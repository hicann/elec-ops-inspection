# Operator PR Checklist

Use this checklist to prepare a contribution that is easy for Electrical Engineering SIG maintainers to review.

## 1. Scenario Fit

- The operator maps to a concrete power equipment inspection workflow.
- The README explains the field problem, such as defect detection, noisy-site speech interaction, 3D reconstruction, point cloud deduplication, route planning, or edge-side data preprocessing.
- The text states why the workload belongs on Ascend NPU instead of CPU-only processing.
- The claimed use case is specific enough to validate, not only a generic AI acceleration statement.

## 2. Operator Interface

- Inputs, outputs, attributes, dtypes, shapes, and data formats are documented.
- Optional outputs and undefined outputs are called out explicitly.
- Shape relationships are written as formulas when needed.
- Unsupported values, edge cases, and hardware limits are documented.
- The public call path is clear, such as Python wrapper, pybind, aclnn, host tiling, and kernel implementation.

## 3. Accuracy Evidence

- The baseline implementation is named, for example PyTorch, torchaudio, NumPy, or a CPU reference.
- Test shapes include normal cases, boundary cases, and at least one realistic inspection-scale workload.
- Tolerance thresholds are justified by dtype.
- For exact operators, outputs that must match exactly are listed.
- For approximate operators, metrics and tolerances are listed.

## 4. Performance Evidence

- Hardware model, CANN version, framework version, and input shape are included.
- Baseline and optimized timing use the same unit.
- Speedup claims are computed from stated measurements.
- The README explains the main source of acceleration, such as HBM traffic reduction, kernel fusion, tiling, double buffering, multi-core balancing, or vectorized mask extraction.
- Small-shape overhead and large-shape scaling behavior are both considered when applicable.

## 5. Repository Hygiene

- New source files are placed under the operator directory that owns them.
- Examples do not require private datasets or local absolute paths.
- Generated binaries, build folders, cache files, local logs, tokens, and credentials are not committed.
- Markdown tables render cleanly and use consistent naming.
- Claims in root `README.md` match the operator README.

## 6. PR Summary

- The PR title names the operator or documentation target.
- The description states the changed files and reviewer-facing value.
- Tests or manual checks are listed with exact commands when possible.
- Known limitations are disclosed.
