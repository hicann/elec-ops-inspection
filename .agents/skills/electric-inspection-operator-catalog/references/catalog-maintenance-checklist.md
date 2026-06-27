# Catalog Maintenance Checklist

Read this reference when updating or reviewing the root `README.md` operator catalog. It expands the checks from `SKILL.md` and provides row templates without duplicating the full workflow.

## Inventory Checks

- The operator appears exactly once in the root operator table.
- The table name matches the directory name or a documented alias.
- The operator directory contains a README or a nearby usage document.
- Deprecated or migrated operators are clearly marked instead of silently removed.
- New rows do not reorder unrelated entries unless the user requested sorting.

## Field Rules

| Field | Rule | Avoid |
| --- | --- | --- |
| 算子名称 | Use a stable code-style name such as `` `unique_v3` ``. Add a link only if the target exists. | Marketing names, undocumented aliases, or names that differ from source directories. |
| 场景 | Use short labels supported by README/operator docs. | Claiming a power inspection scenario from a generic algorithm name alone. |
| 描述 | State the operator function and why it matters for inspection or support workflows. | Benchmark, accuracy, or deployment claims without evidence. |
| 状态 | Use conservative labels: `已发布`, `开发中`, `实验性`, `待补齐`, `已下线`. | Using `已发布` when build/run docs are missing or status is unknown. |

## Scenario Label Guidance

- `CV视觉检测`: object detection, defect detection, image/video preprocessing, feature extraction, or vision post-processing for inspection scenes.
- `具身巡检`: robot dog, drone, 3D reconstruction, 3DGS, point cloud, SLAM, route planning, or edge-side spatial perception.
- `语音识别`: ASR, RNN-T, voice interaction, noisy-site command recognition, or inspection assistant dialogue.
- `数学通用算子`: generic math operators that support inspection pipelines but are not scene-specific.
- `通用支撑`: shared utilities, data movement, format conversion, or infrastructure operators.
- `待确认`: use only in review notes, not as a final README scenario label, unless the maintainer explicitly wants a placeholder.

## Link Checks

For each changed row:

```text
[ ] Linked path exists.
[ ] Link target is relative to repository root.
[ ] Linked README or docs explain the operator at a reviewer-readable level.
[ ] No link points to a local absolute path, private environment, or generated build output.
```

## PR Description Snippet

```markdown
## 变更内容

- 更新根 README 算子清单，补齐/修正 `<operator_name>` 的场景、描述、状态或链接。
- 保持现有表格结构，仅调整与本次算子目录/文档一致性相关的行。

## 依据

- `<operator_name>/README.md`
- `<operator_name>/docs/...`

## 验证

- `git diff --check`
- 已人工核对 README 链接目标存在

## 未验证项

- 未进行性能、精度或硬件兼容性声明；这些证据应由专门 benchmark 或示例复现材料提供。
```

## Reviewer Notes

Flag these for maintainers:

- A README row refers to an operator directory that is absent from the branch.
- A new operator directory exists but has no README or usage document.
- A scenario label depends on external context that is not present in the repository.
- A status label changed from `开发中`/`实验性` to `已发布` without build, usage, or release evidence.
