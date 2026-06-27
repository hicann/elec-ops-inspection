# 算子运行时诊断报告模板

读取时机：已经完成阶段定位或需要整理 Issue、review、PR 说明时。

未实际在目标 NPU/CANN 环境执行的项必须写 `NOT_RUN`。

````markdown
## Operator Runtime Triage

### Scope

- Operator:
- Commit:
- Entry point:
- CANN / PyTorch / torch_npu:
- SoC:
- Current device:
- Minimal input:

### First Observable Failure

- Original command:
- Last successful operation:
- First failing operation:
- Immediate return code:
- Recent error:
- First failing synchronization point:
- Classification:

### Runtime Timeline

| Step | API/operation | Device/stream | Result | Evidence |
|---|---|---|---|---|
| pre-call sync |  |  |  |  |
| dispatcher/entry |  |  |  |  |
| library/symbol |  |  |  |  |
| descriptor |  |  |  |  |
| GetWorkspaceSize |  |  |  |  |
| workspace allocation |  |  |  |  |
| submit |  |  |  |  |
| post-call sync |  |  |  |  |
| result copy/read |  |  |  |  |

### Library and Symbols

| Item | Expected | Actual | Status |
|---|---|---|---|
| custom op path |  |  |  |
| vendor priority |  |  |  |
| op_api library |  |  |  |
| `<op>GetWorkspaceSize` |  |  |  |
| `<op>` |  |  |  |
| dynamic dependencies |  |  |  |

### Tensor Preflight

| Tensor | Shape | Dtype | Device | Stride/contiguous | Descriptor |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

### Workspace and Lifetime

- Workspace size:
- Allocation result:
- Executor:
- Descriptor lifetime:
- Workspace lifetime:
- Output lifetime:

### Device and Stream

| Layer | Device | Stream/context | Evidence |
|---|---|---|---|
| Python current |  |  |  |
| input/output |  |  |  |
| C++ submit |  |  |  |
| synchronize |  |  |  |

### Root Cause or Blocker

- Classification:
- Evidence:
- Why this is the earliest failing stage:
- Downstream errors excluded:
- Missing evidence:

### Minimal Reproduction

```text
working directory:
command:
single-call input:
sync before:
sync after:
expected:
actual:
```

### Validation Status

- Static stage mapping:
- Library/symbol:
- Descriptor/GetWorkspaceSize:
- Workspace:
- Submit:
- Synchronize:
- Result read:
- NPU execution:
- Build/install handoff:
- Precision handoff:
- Approval-required actions:
````

## 填写要求

- 时间线按真实执行顺序填写，不按猜测的根因排序。
- 返回码、recent error 和 plog 只保留首错所需上下文。
- 路径和环境变量必须脱敏，不写入令牌、认证头或业务数据。
- `submit success`、`sync success`、`result readable`、`precision pass` 是四个独立结论。
- 静态代码观察写 `dry_run`，不能写成目标设备已验证。
