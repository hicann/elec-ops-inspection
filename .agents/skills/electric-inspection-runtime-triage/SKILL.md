---
name: electric-inspection-runtime-triage
description: 当 elec-ops-inspection 的 CANN、Ascend C、ACLNN 或 PyTorch NPU 算子已经完成构建安装且 Python 模块可导入，但在自定义 op_api 动态库/符号解析、aclTensor 描述符、GetWorkspaceSize、executor/workspace、设备与流、算子提交、流同步、异步设备错误或结果拷贝阶段失败时使用；按首个可观察失败阶段收集证据并生成最小复现，不用于构建安装、精度偏差或性能回归。
---

# Electric Inspection Runtime Triage

使用本 skill 定位算子“已经能够被调用方加载”之后、结果可安全读取之前的运行时故障。核心原则是先建立同步边界，再区分符号解析、描述符、GetWorkspaceSize、workspace、提交、设备执行和结果拷贝，避免把异步设备错误误归因于后续任意一行 Python。

## 单一职责

本 skill 处理：

- Python 扩展已成功 import，但 dispatcher、自定义 op_api 或 ACLNN 符号不可用。
- `dlopen`、`dlsym`、`libcust_opapi.so`、vendor 加载顺序相关问题。
- `aclCreateTensor`、输入 device/dtype/shape/stride 与描述符创建失败。
- `aclnn*GetWorkspaceSize`、`aclOpExecutor`、workspace 大小或分配失败。
- 当前 NPU device、stream 与输入/输出 tensor 所在设备不一致。
- ACLNN 提交成功或返回失败后的定位。
- 错误只在 `torch.npu.synchronize()`、`aclrtSynchronizeStream()` 或结果拷贝时出现。
- 资源生命周期、错误上下文丢失和最小运行复现。

本 skill 不处理：

- CMake、编译、链接、`.run`、wheel、vendor 安装树或 Python import 失败；移交 Build and Install Triage。
- output、loss、grad、inverse、counts 数值不一致；移交 Precision Debugging。
- benchmark、吞吐、时延、显存或性能回归。
- 未经授权修改设备环境、重装 CANN、清理 vendor 或终止其他 NPU 任务。

## 进入条件

开始前至少确认：

1. 目标 Python 模块或共享库能够加载。
2. 错误发生在算子调用、同步或结果读取路径。
3. 能提供原始命令、目标 commit 和第一段完整异常。

若 import 尚未通过，停止运行时诊断并分类为 `build_install_handoff`。不要因为异常文本出现 `.so` 就跳过加载阶段判断。

## 输入要求

收集：

- 算子、commit、调用入口、最小输入 shape/dtype/device/stride。
- Python、PyTorch、torch_npu、CANN 版本和目标 SoC。
- 当前 device、输入 tensor device、当前 stream 的可见信息。
- 原始调用栈、ACL 返回码、`aclGetRecentErrMsg()` 或 plog 中对应时间段。
- 调用前后是否有显式同步，以及错误首次出现在哪个同步点。
- `ASCEND_CUSTOM_OPP_PATH`、`ASCEND_OPP_PATH`、`LD_LIBRARY_PATH` 的脱敏解析结果。
- 预期加载的库名、符号名和实际解析结果。

不要收集令牌、认证头、完整私有环境变量或业务输入数据。路径只保留诊断所需部分。

## 运行时阶段模型

| 阶段 | 成功证据 | 典型失败 |
|---|---|---|
| 0. 调用入口 | import 成功，调用到达 C++/dispatcher | schema/dispatcher 未注册、入口参数绑定失败 |
| 1. 库与符号 | 目标 op_api 库已加载，两个 ACLNN 符号可解析 | vendor 路径、`dlopen`、`dlsym`、符号版本问题 |
| 2. 描述符与预检 | device/dtype/shape/stride 合法，描述符非空 | device 不一致、不支持 dtype、空描述符 |
| 3. GetWorkspaceSize | 返回成功，workspace size 与 executor 有记录 | 参数校验、算子注册、tiling 或 executor 创建失败 |
| 4. Workspace | 零 workspace 合法，非零分配成功且生命周期覆盖提交 | OOM、大小/类型错误、过早释放 |
| 5. 提交 | ACLNN 调用返回成功，提交到预期 stream | 立即返回错误、stream/context 不匹配 |
| 6. 同步 | 紧邻调用的 stream/device 同步成功 | 异步 Kernel、tiling、越界或设备错误 |
| 7. 结果读取 | device-to-host 或 tensor 访问成功 | 拷贝方向/大小、失效输出、延迟错误 |

只有上一阶段有证据通过，才把故障归到下一阶段。完整检查项见 [运行时阶段矩阵](references/runtime-stage-matrix.md)。

## 工作流

### Step 1：冻结首个可观察错误

记录：

- 原始命令和工作目录。
- 算子调用前最后一个成功操作。
- 第一个抛异常或返回非零的位置。
- 第一个失败同步点。
- 结果读取是否发生。

保留原始失败日志后再增加诊断同步。不要先循环重试、换随机输入或清理日志；这些动作会破坏首错顺序。

输出：原始时间线。

### Step 2：建立最小同步边界

异步执行下，Python 报错行不一定是设备首错行。构造单次调用：

```python
torch.npu.synchronize()
result = custom_op(...)
torch.npu.synchronize()
```

若使用直接 ACL 接口，则在目标 stream 上调用 `aclrtSynchronizeStream(stream)`。不要用无关 device 或无关 stream 的同步结果证明目标调用成功。

分类：

- 调用立即失败：继续检查阶段 0 至 5。
- 调用返回、紧邻同步失败：`async_device_error`。
- 紧邻同步通过、结果读取失败：进入阶段 7。
- 无法增加同步：`blocked_missing_sync`，不猜设备首错。

输出：调用、同步、读取三点时间线。

### Step 3：确认是否应移交构建安装

确认：

- Python extension/import 已通过。
- 依赖库解析已通过到足以进入算子调用。
- 当前错误不是 extension `.so` 本身无法加载。

如果自定义 op_api 是在运行时按 vendor 路径动态发现，库搜索和 ACLNN 符号解析仍属于本 skill；如果 Python 扩展本身无法加载，则属于构建安装。

输出：`runtime_scope_confirmed` 或 `build_install_handoff`。

### Step 4：检查库搜索与符号解析

记录：

- 实际搜索的 `ASCEND_CUSTOM_OPP_PATH` 条目。
- `$ASCEND_OPP_PATH/vendors/config.ini` 的 `load_priority` 顺序。
- 目标库是 `libcust_opapi.so` 还是默认 `libopapi.so`。
- `<op>GetWorkspaceSize` 与 `<op>` 两个符号是否都存在。
- `dlopen`/`dlsym` 的原始错误。

可使用只读命令：

```bash
ldd <extension-or-op-api-library>
readelf -d <library>
nm -D <op-api-library> | rg "aclnn.*(GetWorkspaceSize)?"
```

不要仅凭文件存在判定加载成功，也不要把修改 `LD_LIBRARY_PATH` 当作所有符号问题的通用修复。

输出：库搜索顺序、命中库、目标符号和失败点。

### Step 5：检查输入预检与 ACL 描述符

在调用 ACLNN 前记录：

- 每个输入的 shape、dtype、device、stride、contiguous 状态。
- 输出分配的 shape、dtype、device。
- 当前 device 与所有 tensor device。
- dtype 转换是否显式拒绝不支持类型。
- `aclCreateTensor` 等描述符创建结果是否检查为空。

可见路径缺少检查时分类为 `preflight_gap`，仅说明诊断保护不足；没有运行证据时不要写成已发生的设备故障。

输出：输入/输出预检表和描述符状态。

### Step 6：隔离 GetWorkspaceSize 与 executor

记录：

- 完整 API 名。
- 返回码和 recent error。
- workspace size。
- executor 是否有效。
- 所有标量参数和输入摘要。

`GetWorkspaceSize` 失败常常是最早的参数、注册或 tiling 失败点。错误信息若只保留通用字符串而丢失返回码/recent error，分类为 `diagnostic_context_loss`，先改进证据收集，再决定根因。

输出：GetWorkspaceSize 证据卡。

### Step 7：验证 workspace 与生命周期

检查：

- `workspace_size == 0` 是否走空指针合法路径。
- 非零 workspace 的分配设备、字节数和对齐。
- 分配对象是否存活到提交完成。
- executor、描述符、输入、输出和转换参数的生命周期。
- OOM 是否发生在 workspace，而不是输出或其他缓存分配。

不要仅根据“使用框架 allocator”推断生命周期一定正确；绑定实际 stream、捕获对象和释放位置。

输出：workspace/lifetime 表。

### Step 8：核对 device、context 与 stream

对照：

- Python 当前 device。
- 输入和输出 tensor device。
- C++ wrapper 获取 stream 时使用的 device。
- 直接 ACL 样例创建的 context/stream。
- 同步时使用的 device/stream。

源码中出现固定设备编号但运行环境信息不足时，分类为 `device_stream_needs_confirmation`。只有复现证明目标 tensor 与 stream 不一致，才能升级为 `runtime_confirmed`。

输出：device/stream 对照表。

### Step 9：区分提交错误与异步设备错误

记录两类信号：

1. ACLNN 提交函数立即返回的状态和 recent error。
2. 紧邻目标调用的同步结果和设备日志。

提交返回成功只表示任务已进入执行路径，不等于 Kernel 已成功完成。同步失败时保留：

- 第一个设备错误码。
- 目标 stream。
- 算子名和输入摘要。
- plog 时间窗口。
- 是否能用单次调用稳定复现。

输出：`submit_failed` 或 `async_device_error`，不要混为一类。

### Step 10：检查结果读取与清理

同步通过后再检查：

- 输出 shape/dtype/device 和有效长度。
- device-to-host 拷贝方向、字节数和目标容量。
- 返回 tensor 是否仍引用有效存储。
- 描述符、workspace、executor 和转换参数的清理顺序。

数值能够读取但不正确时，停止并移交 Precision Debugging。

输出：结果读取状态和移交项。

### Step 11：生成最小复现与报告

最小复现应：

- 单进程、单 device、单 stream、单次调用。
- 固定 shape/dtype 和必要标量。
- 调用前后显式同步。
- 打印阶段标签、返回码和脱敏环境摘要。
- 不包含 benchmark 循环、随机业务数据或其他算子。

使用 [运行时诊断报告模板](references/runtime-report-template.md) 输出可直接放入 Issue/PR 的报告。

## 仓库内算子观察点

### `optimized_transducer`

- pybind wrapper 直接创建 ACL tensor，调用 `aclnnOptimizedTransducerGetWorkspaceSize` 后分配 workspace，再获取 NPU stream 提交。
- wrapper 当前通用异常文本不保留返回码或 recent error，静态分类为 `diagnostic_context_loss`。
- dtype 转换存在默认分支，描述符返回值未见逐项预检；静态分类为 `preflight_gap`，不是已确认运行缺陷。
- wrapper 获取 stream 时传入固定设备编号 `0`；在多卡或非零当前设备场景需分类为 `device_stream_needs_confirmation`。
- Python 样例在调用前后同步；直接 ACL 样例也在提交后同步，再执行 device-to-host 拷贝，可用作阶段边界参考。

### `unique_v3`

- cpp_extension helper 会从 `ASCEND_CUSTOM_OPP_PATH` 和 vendor `load_priority` 构造自定义 op_api 搜索路径。
- `EXEC_NPU_CMD` 同时解析 `aclnnUniqueV3GetWorkspaceSize` 与 `aclnnUniqueV3`，并在失败信息中保留 recent error。
- 提交通过 `OpCommand` 路径完成；Python 测试在调用前后同步，因此需分别记录 submit 与 synchronize 结果。
- 静态可确认的是“哪些诊断信号存在”；没有 NPU 运行日志时，不声称库、Kernel 或结果已经验证通过。

这些观察点用于 dry-run 和定位入口，不是缺陷清单。

## 分类

- `runtime_scope_confirmed`
- `build_install_handoff`
- `dispatcher_registration_failed`
- `symbol_resolution_failed`
- `preflight_gap`
- `descriptor_creation_failed`
- `getworkspace_failed`
- `workspace_allocation_failed`
- `submit_failed`
- `async_device_error`
- `device_stream_needs_confirmation`
- `diagnostic_context_loss`
- `result_copy_failed`
- `blocked_missing_sync`
- `runtime_confirmed`
- `precision_handoff`

每个结论必须绑定调用、返回码、同步点、路径或日志证据。

## 错误处理

- 无 NPU/CANN 环境：只做静态阶段映射和 observability dry-run，运行状态写 `NOT_RUN`。
- 只有日志尾部：请求首错附近日志和单次复现，不从连锁错误推断根因。
- recent error 为空：同时保留返回码、API 名和 plog 时间窗口。
- 多次调用后才失败：先降为单次，再逐步恢复循环；不要直接称为内存泄漏。
- 设备处于错误状态：停止追加调用，保留日志；重置设备或进程前请求用户授权并说明影响。
- 需要修改系统环境或安装包：移交构建安装流程并请求授权。

## 验证

至少执行：

```bash
git diff --check
rg -n "GetWorkspaceSize|aclGetRecentErrMsg|aclCreateTensor|getCurrentNPUStream|Synchronize|EXEC_NPU_CMD|dlopen|dlsym" optimized_transducer unique_v3
rg -n "(A[K]IA|BEGIN [A-Z ]*PRIVATE K[E]Y|GITCODE[_]TOKEN|Authorizatio[n]:|passwor[d]\\s*=|t[o]ken\\s*=|s[e]cret\\s*=)" <changed-paths>
```

对两个现有算子完成静态 dry-run，至少给出：

- 库/符号、GetWorkspaceSize、提交、同步、读取的阶段映射。
- 一项已有诊断信号和一项诊断上下文缺口。
- device/stream 是否需要运行确认。
- 所有未实际执行项的 `NOT_RUN` 标记。

## 输出格式

最终返回：

1. 原始错误时间线与首个可观察失败点。
2. 库、符号、描述符、GetWorkspaceSize、workspace、submit、sync、copy 阶段状态。
3. 输入、device、stream 和同步边界。
4. 返回码、recent error 与日志证据。
5. 根因或阻塞分类及最小复现。
6. 已验证、`NOT_RUN`、需授权动作和后续移交。

## 参考资料

- [运行时阶段矩阵](references/runtime-stage-matrix.md)
- [运行时诊断报告模板](references/runtime-report-template.md)
