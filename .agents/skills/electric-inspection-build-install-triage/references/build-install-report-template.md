# 构建安装诊断报告模板

读取时机：完成首错定位、产物检查或安装验收后，用于整理 review、Issue 或 PR 说明。

未执行的编译、安装或 import 必须写 `NOT_RUN`。

````markdown
## Build and Install Triage

### Scope

- Operator:
- Commit:
- Host architecture:
- Working directory:
- Original command:
- Exit code:

### First Failure

- Stage:
- First error:
- Last successful stage:
- Last verified artifact:
- Classification:

### Environment

| Item | Version/path status | Evidence |
|---|---|---|
| CANN |  |  |
| CMake/compiler |  |  |
| Python/pip |  |  |
| PyTorch/torch_npu |  |  |
| vendor |  |  |

### Entrypoints

| Command path | Exists | Expected cwd | Source |
|---|---|---|---|
|  |  |  | repository / external |

### Artifacts

| Stage | Expected | Actual | Status |
|---|---|---|---|
| configure | `CMakeCache.txt` |  |  |
| compile |  |  |  |
| package | `.run` / wheel |  |  |
| install | vendor/Python tree |  |  |
| import | module and dependencies |  |  |

### Root Cause Evidence

- Evidence:
- Why this is the first cause:
- Downstream errors:

### Minimal Fix

- Change:
- Rerun command:
- Expected success artifact:
- Destructive/system write approval required:

### Validation Status

- Static checks:
- Configure:
- Compile:
- Package:
- Install:
- Import:
- Runtime handoff:
- Unverified items:
````

## 填写要求

- 日志只保留首错和必要上下文，不粘贴大量重复输出。
- 命令必须标明工作目录。
- 修复建议只针对第一个失败阶段。
- 不把 import 成功写成算子运行通过。
- 不记录令牌、认证头、私有数据或不可公开的内部路径。
