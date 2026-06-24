# Benchmark Report Template

Read this file when drafting benchmark sections for an operator README, PR body, or review response.

```markdown
## Benchmark Summary

- Operator:
- Inspection scenario:
- Benchmark purpose:
- Status: measured / dry-run / pending hardware validation

## Environment

| Item | Value |
|---|---|
| Ascend device |  |
| CANN version |  |
| Framework/runtime |  |
| Baseline environment |  |
| Measurement tool |  |

## Workload Matrix

| Case | Shape / parameters | Dtype | Baseline | Optimized | Unit | Speedup | Notes |
|---|---|---|---:|---:|---|---:|---|
|  |  |  |  |  |  |  |  |

## Correctness Guardrail

- Reference:
- Threshold or exact-match fields:
- Command:
- Result:

## Timing Protocol

- Warmup:
- Repeats:
- Synchronization:
- Statistic:

## Interpretation

- Safe claim:
- Limitation:
- Follow-up validation:
```

Keep unsupported values blank or mark them as `pending`. Do not write placeholders as measured results.
