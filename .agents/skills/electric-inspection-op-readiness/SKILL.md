---
name: electric-inspection-op-readiness
description: Use when preparing, reviewing, or polishing an elec-ops-inspection operator contribution for power equipment inspection scenarios, especially CANN/Ascend C custom operators that need business-scenario framing, interface constraints, accuracy evidence, performance evidence, and PR-ready documentation.
---

# Electric Inspection Operator Readiness

Use this skill before submitting a new operator, operator README update, or review response in `cann/elec-ops-inspection`.

## Scope

This skill focuses on operator contributions for Electrical Engineering SIG inspection scenarios:

- CV visual inspection of transmission corridors, substations, distribution devices, and typical defects.
- Embodied inspection with drones, robot dogs, 3D reconstruction, point cloud processing, speech interaction, or edge-side AI agents.
- CANN/Ascend C operator implementations, PyTorch extension wrappers, aclnn interfaces, host tiling, kernel scheduling, and packaging docs.

Do not use this skill to invent benchmark numbers, claim unsupported hardware compatibility, or rewrite unrelated repository structure.

## Workflow

1. Identify the contribution type:
   - New operator implementation.
   - Operator documentation or scenario expansion.
   - Accuracy/performance evidence update.
   - Review cleanup before PR submission.
2. Read the nearest operator files:
   - Root `README.md` for project positioning and current operator table.
   - The operator `README.md` for scenario, interface, constraints, accuracy, and performance sections.
   - `op_host/`, `op_kernel/`, `pybind/`, `examples/`, or `docs/` files touched by the contribution.
3. Check the contribution against `references/operator-pr-checklist.md`.
4. Produce a concise readiness report using `references/pr-description-template.md`.
5. If gaps remain, fix only the files in scope or list the missing evidence clearly.

## Required Review Points

- Power inspection value: the operator must explain which inspection workflow it supports and why Ascend-side acceleration matters.
- Interface clarity: inputs, outputs, attributes, shapes, dtypes, layouts, and constraints must be stated.
- Engineering path: README or docs should name the Python, aclnn, host, and kernel entry points that users can inspect.
- Accuracy evidence: claims should include reference implementation, dtype, shape range, threshold, and pass/fail result.
- Performance evidence: claims should include hardware, software stack, workload shape, baseline, measurement unit, and acceleration ratio.
- Reproducibility: examples or tests should be runnable from documented commands and avoid hidden local paths.
- Compliance: do not add generated build artifacts, credentials, local logs, or unlicensed third-party files.

## Output Format

When reporting readiness, use:

```markdown
## PR Readiness

- Scenario fit:
- Interface/docs:
- Accuracy evidence:
- Performance evidence:
- Reproducibility:
- Risks or missing items:
```

Keep the report factual. Separate verified facts from assumptions.
