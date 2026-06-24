# Shape 边界报告模板

读取时机：完成 Shape/Tiling 静态分析或运行边界用例后，用于整理 review 结论、PR 描述或缺陷复现。

未运行 NPU 的用例写 `NOT_RUN`；静态可达风险写 `unguarded_static_risk`，不要升级为运行缺陷。

````markdown
## Shape Boundary Analysis

### Scope

- Operator:
- Commit or diff:
- Entry point:
- Files inspected:

### Shape Evidence Chain

| Layer | Declared shape/rank | Evidence | Status |
|---|---|---|---|
| README/wrapper |  |  |  |
| OpDef/infer |  |  |  |
| Host Tiling |  |  |  |
| Kernel |  |  |  |
| Example |  |  |  |

### Derived Formulas

| Variable | Formula | Type | Unit | Preconditions | Consumer |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

### Transition Points

| ID | Input | Derived values | Expected path | Rationale |
|---|---|---|---|---|
| TILE-001 |  |  |  |  |

### Relational Constraints

| ID | Constraint | Minimal valid | Minimal invalid | Guard | Status |
|---|---|---|---|---|---|
| REL-001 |  |  |  |  |  |

### Findings

| Severity | Classification | Input/condition | Code path | Impact | Evidence |
|---|---|---|---|---|---|
|  | guarded / derived_valid / contract_conflict / unguarded_static_risk / runtime_confirmed / not_covered |  |  |  |  |

### Boundary Cases

| Case ID | Shape and attributes | Formula result | Expected | Run status | Evidence |
|---|---|---|---|---|---|
| ZERO-001 |  |  |  | NOT_RUN |  |

### Commands

```text
<static checks and runtime commands>
```

### Conclusion

- Guarded:
- Derived valid:
- Contract conflicts:
- Unguarded static risks:
- Runtime confirmed:
- NPU-only validation:
- Unverified items:
````

## 填写要求

- 代码路径精确到文件和行号。
- 公式注明单位，尤其是元素数、字节数、tile 数和 block 数。
- 每个 finding 绑定一个最小输入或符号条件。
- 不把 README 约束自动视为 Host 已实现保护。
- 不把静态风险写成实际崩溃、越界或 OOM。
