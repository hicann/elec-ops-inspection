---
name: inspection-benchmark-evidence
description: Use when preparing, reviewing, or updating benchmark evidence for elec-ops-inspection operator contributions, especially Ascend/CANN performance reports that need hardware/software environment, workload shape, baseline, measurement protocol, statistics, reproducibility commands, and reviewer-ready claims.
---

# Inspection Benchmark Evidence

Use this skill when an `elec-ops-inspection` contribution needs benchmark evidence for a new operator, an operator README, a PR description, or a review response.

This skill is about benchmark evidence only. It is not a general PR readiness checklist and it is not a skill-creation workflow.

## Scope

Apply this skill to performance evidence for power equipment inspection operators, including:

- CV inspection operators for defect detection, transmission corridor analysis, and substation equipment recognition.
- Embodied inspection operators for drones, robot dogs, 3D reconstruction, point-cloud preprocessing, sparse graph construction, and edge-side planning.
- Voice-agent related operators for noisy-site inspection interaction, ASR training, RNN-T loss, softmax fusion, or memory-pressure reduction.

Do not invent benchmark numbers, hardware compatibility, baseline results, workload distributions, or field scenarios. If evidence is missing, mark it as missing.

## Required Inputs

Before making benchmark claims, collect:

- Operator path and public entry points, such as README, Python wrapper, aclnn entry, host tiling, kernel, examples, and scripts.
- Hardware and software stack: Ascend model, CANN version, driver/firmware when available, framework version, CPU/GPU baseline environment when used.
- Workload matrix: shapes, dtypes, layouts, batch sizes, sequence lengths, point counts, vocabulary size, or other operator-specific dimensions.
- Baseline definition: CPU reference, framework implementation, previous kernel, or external accelerator baseline.
- Measurement protocol: warmup count, repeat count, timing API, synchronization point, statistics, and unit.
- Accuracy guardrail: reference output, threshold, exact-match fields, or correctness command used before timing.

## Workflow

1. Read the target context:
   - Root `README.md` for repository positioning and operator list.
   - The operator `README.md` for current accuracy and performance sections.
   - Nearby `examples/`, `pybind/`, `op_host/`, `op_kernel/`, and scripts that define how the operator is built or invoked.
2. Classify the benchmark scenario using `references/scenario-matrix.md`.
3. Check the available evidence against `references/evidence-checklist.md`.
4. Draft or review the benchmark report using `references/report-template.md`.
5. Separate claims into:
   - Verified in the current evidence.
   - Derived from stated measurements.
   - Not yet verified and requiring follow-up.
6. Before PR submission, ensure the report includes exact commands or explains why the environment cannot run them.

## Claim Rules

- Use measured values only when the source includes environment, workload, unit, and baseline.
- Compute speedup only from stated numbers and keep the formula inspectable.
- Do not compare CPU, GPU, and NPU results without naming the hardware and software stack for each.
- Do not use field terms such as "real-time", "ms-level", "minute-level", or "production-ready" unless the measured workload and latency target are stated.
- For documentation-only changes, state that no new benchmark was executed.

## Error Handling

- Missing hardware: mark the report as `pending hardware validation` and keep performance numbers out of the final claim.
- Missing baseline: report optimized timing only as an isolated measurement and do not compute speedup.
- Missing workload shape: stop the benchmark summary and request the exact shape, dtype, and layout.
- Mixed units or statistics: normalize the table before comparing results, or split the rows into separate tables.
- Failed correctness check: treat timing as non-actionable until correctness is fixed or the failure is explained.

## Output Format

```markdown
## Benchmark Evidence Review

- Operator:
- Scenario:
- Verified evidence:
- Missing evidence:
- Claims safe to keep:
- Claims to weaken or remove:
- Reproducibility commands:
- Reviewer notes:
```

## References

- `references/evidence-checklist.md`: benchmark evidence checklist.
- `references/report-template.md`: PR or README benchmark report template.
- `references/scenario-matrix.md`: inspection scenario to benchmark dimension mapping.
