# Benchmark Evidence Checklist

Read this file when checking whether an operator benchmark claim is ready for a README or PR description.

## Environment

- Ascend hardware model is named, for example Atlas 910B or another verified target.
- CANN version is listed.
- Framework version is listed when using PyTorch, torch_npu, torchaudio, NumPy, or a custom wrapper.
- CPU or GPU baseline hardware and software are listed when used for comparison.
- Power mode, device count, and isolation assumptions are recorded when they affect the result.

## Workload

- Shape dimensions are explicit and use operator-specific names.
- Dtypes and data layout are recorded.
- Input distribution is described when it affects timing, such as repeated values for `unique_v3` or vocabulary size for RNN-T.
- Batch size and sequence or point-count ranges cover the intended inspection scenario.
- Boundary cases are separated from typical cases.

## Correctness Before Timing

- A reference implementation or prior validated output is named.
- Accuracy threshold is stated for approximate operators.
- Exact-match fields are stated for exact operators.
- The correctness command or comparison script is recorded.
- Failed or skipped correctness checks are not hidden.

## Timing Protocol

- Warmup count and measured repeat count are listed.
- The timing API is named.
- Device synchronization is performed before reading elapsed time.
- Reported statistic is stated, such as mean, median, p50, p90, min, or max.
- Unit is consistent across baseline and optimized measurements.

## Claim Hygiene

- Speedup is computed from the same workload and unit.
- Small-shape overhead is not hidden when kernel launch or data transfer dominates.
- Large-shape scaling is described only when enough shape points are measured.
- A benchmark table does not mix training, inference, and preprocessing results without labeling them.
- Claims that are not measured are written as required validation items, not results.
