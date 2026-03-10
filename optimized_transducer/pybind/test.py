import torch
import torchaudio
import torch_npu
import time
import warnings
warnings.filterwarnings("ignore")
import optimized_transducer_ascend_ops as ops

torch_npu.npu.set_device(0)

def run_test(B, T, U, V, warmup=False):
    if not warmup:
        print(f"测试  B={B}, T={T}, U={U}, V={V}")
    torch.manual_seed(42)
    
    # 构造输入
    logits_4d = torch.randn(B, T, U, V, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(1, V, (B, U - 1), dtype=torch.int32)
    logit_lens = torch.full((B,), T, dtype=torch.int32)
    target_lens = torch.full((B,), U - 1, dtype=torch.int32)
    logits_packed = logits_4d.view(B * T * U, V)

    # CPU 
    loss_cpu, grad_cpu, cpu_time = None, None, 0.0
    start = time.perf_counter()
    loss_cpu = torchaudio.functional.rnnt_loss(
        logits_4d, targets.int(), logit_lens.int(), target_lens.int(),
        blank=0, reduction='none', fused_log_softmax=True
    )
    loss_cpu.sum().backward()
    cpu_time = time.perf_counter() - start
    grad_cpu = logits_4d.grad.clone()

    # NPU
    logits_npu = logits_packed.to("npu:0")
    targets_npu = targets.to("npu:0")
    logit_lens_npu = logit_lens.to("npu:0")
    target_lens_npu = target_lens.to("npu:0")
    torch_npu.npu.synchronize()

    # log_softmax
    log_softmax_cpu = torch.nn.functional.log_softmax(logits_4d, dim=-1)

    # 计时
    start = time.perf_counter()
    loss_npu, grad_npu = ops.rnnt_loss(
        logits_npu, targets_npu, logit_lens_npu, target_lens_npu, 0
    )
    torch_npu.npu.synchronize()
    npu_time = time.perf_counter() - start

    if warmup:
        return

    # 对比结果
    loss_diff = (loss_npu.cpu() - loss_cpu.detach()).abs()
    grad_diff = (grad_npu.cpu().view(B, T, U, V) - grad_cpu).abs()
    log_softmax_diff = (grad_npu.cpu().view(B, T, U, V) - log_softmax_cpu).abs()

    print("=" * 60)
    print("性能")
    print(f"  CPU 耗时 : {cpu_time*1000:.2f} ms")
    print(f"  NPU 耗时 : {npu_time*1000:.2f} ms")
    print(f"  加速比   : {cpu_time/npu_time:.2f}")
    print("-" * 60)
    print("精度")
    print(f"  Loss 绝对误差 : Max = {loss_diff.max():.2e} | Mean = {loss_diff.mean():.2e}")
    print(f"  LogSoftmax 绝对误差: Max = {log_softmax_diff.max():.2e} | Mean = {log_softmax_diff.mean():.2e}")
    print(f"  Grad 绝对误差 : Max = {grad_diff.max():.2e} | Mean = {grad_diff.mean():.2e}")
    print("=" * 60 + "\n ")


# 预热
print(">>> Warmup ...")
run_test(300, 10, 10, 10, warmup=True)

# 测试
print(">>> Testing ...")
run_test(10, 10, 10, 1000)

run_test(100, 10, 10, 1000)

run_test(1, 10, 10, 2000000)

run_test(1, 10000, 1000, 10)

run_test(56, 5678, 648, 2304)