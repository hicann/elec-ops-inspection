# Copyright 2026 Electrical Engineering SIG - CANN Community
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import time

import torch
import torchaudio
import torch_npu
import optimized_transducer_ascend_ops as ops


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare optimized_transducer NPU results with torchaudio."
    )
    parser.add_argument("--device", type=nonnegative_int, default=0)
    parser.add_argument("--batch", type=positive_int, default=2)
    parser.add_argument("--time-steps", type=positive_int, default=10)
    parser.add_argument("--target-steps", type=positive_int, default=5)
    parser.add_argument("--vocab-size", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iterations", type=nonnegative_int, default=2)
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=1e-4)
    return parser.parse_args()


def build_inputs(batch, time_steps, target_steps, vocab_size, seed):
    torch.manual_seed(seed)
    logits_4d = torch.randn(
        batch,
        time_steps,
        target_steps,
        vocab_size,
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.randint(
        1, vocab_size, (batch, target_steps - 1), dtype=torch.int32
    )
    logit_lens = torch.full((batch,), time_steps, dtype=torch.int32)
    target_lens = torch.full((batch,), target_steps - 1, dtype=torch.int32)
    return logits_4d, targets, logit_lens, target_lens


def run_cpu_reference(logits_4d, targets, logit_lens, target_lens):
    start = time.perf_counter()
    loss_cpu = torchaudio.functional.rnnt_loss(
        logits_4d,
        targets,
        logit_lens,
        target_lens,
        blank=0,
        reduction="none",
        fused_log_softmax=True,
    )
    loss_cpu.sum().backward()
    cpu_time = time.perf_counter() - start
    return loss_cpu.detach(), logits_4d.grad.detach().clone(), cpu_time


def run_npu(
    logits_4d,
    targets,
    logit_lens,
    target_lens,
    device,
    warmup_iterations,
):
    npu_device = f"npu:{device}"
    logits_npu = logits_4d.detach().view(-1, logits_4d.shape[-1]).to(npu_device)
    targets_npu = targets.to(npu_device)
    logit_lens_npu = logit_lens.to(npu_device)
    target_lens_npu = target_lens.to(npu_device)

    for _ in range(warmup_iterations):
        ops.rnnt_loss(
            logits_npu, targets_npu, logit_lens_npu, target_lens_npu, 0
        )
    torch_npu.npu.synchronize()

    start = time.perf_counter()
    loss_npu, grad_npu = ops.rnnt_loss(
        logits_npu, targets_npu, logit_lens_npu, target_lens_npu, 0
    )
    torch_npu.npu.synchronize()
    npu_time = time.perf_counter() - start
    return (
        loss_npu.cpu(),
        grad_npu.cpu().view_as(logits_4d),
        npu_time,
    )


def validate_accuracy(
    loss_npu,
    grad_npu,
    loss_cpu,
    grad_cpu,
    loss_atol,
    grad_atol,
):
    if loss_npu.shape != loss_cpu.shape:
        raise AssertionError(
            f"loss shape mismatch: NPU={tuple(loss_npu.shape)}, "
            f"CPU={tuple(loss_cpu.shape)}"
        )
    if grad_npu.shape != grad_cpu.shape:
        raise AssertionError(
            f"grad shape mismatch: NPU={tuple(grad_npu.shape)}, "
            f"CPU={tuple(grad_cpu.shape)}"
        )

    if not torch.isfinite(loss_cpu).all().item():
        raise AssertionError("CPU reference loss contains NaN or Inf")
    if not torch.isfinite(grad_cpu).all().item():
        raise AssertionError("CPU reference grad contains NaN or Inf")
    if not torch.isfinite(loss_npu).all().item():
        raise AssertionError("NPU loss contains NaN or Inf")
    if not torch.isfinite(grad_npu).all().item():
        raise AssertionError("NPU grad contains NaN or Inf")

    loss_diff = (loss_npu - loss_cpu).abs()
    grad_diff = (grad_npu - grad_cpu).abs()
    metrics = {
        "loss_max": loss_diff.max().item(),
        "loss_mean": loss_diff.mean().item(),
        "grad_max": grad_diff.max().item(),
        "grad_mean": grad_diff.mean().item(),
    }

    failures = []
    if metrics["loss_max"] > loss_atol:
        failures.append(
            f"loss max diff {metrics['loss_max']:.3e} > {loss_atol:.3e}"
        )
    if metrics["grad_max"] > grad_atol:
        failures.append(
            f"grad max diff {metrics['grad_max']:.3e} > {grad_atol:.3e}"
        )
    if failures:
        raise AssertionError("; ".join(failures))
    return metrics


def main():
    args = parse_args()
    if args.target_steps < 2:
        raise ValueError("--target-steps must be at least 2")
    if args.vocab_size < 2:
        raise ValueError("--vocab-size must be at least 2")
    if (
        not math.isfinite(args.loss_atol)
        or not math.isfinite(args.grad_atol)
        or args.loss_atol < 0
        or args.grad_atol < 0
    ):
        raise ValueError("accuracy tolerances must be finite and nonnegative")

    torch_npu.npu.set_device(args.device)
    print(
        "Test configuration: "
        f"B={args.batch}, T={args.time_steps}, U={args.target_steps}, "
        f"V={args.vocab_size}, device=npu:{args.device}, seed={args.seed}"
    )

    inputs = build_inputs(
        args.batch,
        args.time_steps,
        args.target_steps,
        args.vocab_size,
        args.seed,
    )
    loss_cpu, grad_cpu, cpu_time = run_cpu_reference(*inputs)
    loss_npu, grad_npu, npu_time = run_npu(
        *inputs, args.device, args.warmup_iterations
    )
    metrics = validate_accuracy(
        loss_npu,
        grad_npu,
        loss_cpu,
        grad_cpu,
        args.loss_atol,
        args.grad_atol,
    )

    print("=" * 60)
    print(f"CPU time: {cpu_time * 1000:.2f} ms")
    print(f"NPU time: {npu_time * 1000:.2f} ms")
    print(
        f"Loss absolute error: max={metrics['loss_max']:.3e}, "
        f"mean={metrics['loss_mean']:.3e}"
    )
    print(
        f"Grad absolute error: max={metrics['grad_max']:.3e}, "
        f"mean={metrics['grad_mean']:.3e}"
    )
    print("PASS")


if __name__ == "__main__":
    main()
