# Tiling 审查报告模板

读取时机：完成 TilingData、调度或 workspace 审查后，用于整理 review 结论、PR 描述或缺陷复现记录。

静态容量疑问不能写成已确认越界；没有 NPU 时运行状态写 `NOT_RUN`。

````markdown
## Tiling Review

### Scope

- Operator:
- Commit or diff:
- Host Tiling:
- TilingData:
- Kernel entry and Init:

### Field Lifecycle

| Field | Type/unit | Host assignment | Kernel consumer | Behavior effect | Status |
|---|---|---|---|---|---|
|  |  |  |  |  | closed_loop / dead_or_reserved_field / not_covered |

### Scheduling

- `SetBlockDim`:
- Kernel block count source:
- Work partition formula:
- Coverage/remainder result:
- Synchronization:
- Tiling key mapping:

### Workspace Map

| Region | Start bytes | Length bytes | End bytes | Writer/reader | Evidence |
|---|---:|---:|---:|---|---|
|  |  |  |  |  |  |

- Host custom budget:
- Kernel maximum end:
- Difference:
- System workspace treatment:
- Classification:

### Constants and Units

| Concept | Sources | Current values | Unit | Status |
|---|---|---|---|---|
|  |  |  |  | duplicated_source_of_truth / closed_loop |

### Findings

| Severity | Classification | Evidence | Impact | Minimal recommendation |
|---|---|---|---|---|
|  |  |  |  |  |

### Validation

- Static commands:
- Formula dry-run:
- Runtime command:
- Runtime status:

### Conclusion

- Closed loops:
- Dead or reserved fields:
- Duplicated sources:
- Layout/capacity questions:
- Confirmed static mismatches:
- Runtime confirmed:
- Not covered:
- NPU-only follow-up:
````

## 填写要求

- 所有 workspace 数值统一使用 bytes，必要时附元素数。
- 字段结论必须同时包含 Host 和 Kernel 证据。
- tiling key 缺少构建分发证据时写 `not_covered`。
- `layout_capacity_needs_confirmation` 必须给出公式、代表性输入和差值。
- 只有实际运行复现后才能使用 `runtime_confirmed`。
