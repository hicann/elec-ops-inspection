# 算子运行时阶段矩阵

读取时机：模块已能加载，但自定义算子在调用、同步或结果读取时失败。

本文件用于把错误固定在最早可观察阶段，并选择下一项最小检查。核心职责和操作边界以 `SKILL.md` 为准。

## 导航

- [阶段检查矩阵](#阶段检查矩阵)
- [库与符号证据](#库与符号证据)
- [输入与描述符](#输入与描述符)
- [Workspace 与生命周期](#workspace-与生命周期)
- [Device 与 Stream](#device-与-stream)
- [异步错误协议](#异步错误协议)
- [仓库 dry-run](#仓库-dry-run)

## 阶段检查矩阵

| 阶段 | 必要证据 | 成功判据 | 失败分类 | 下一步 |
|---|---|---|---|---|
| 入口 | import、schema、调用栈 | 到达 C++/ACLNN wrapper | `dispatcher_registration_failed` | 检查注册名和参数绑定 |
| 库 | 搜索路径、命中库、`dlopen` | 目标库 handle 有效 | `symbol_resolution_failed` | 检查 vendor 顺序与依赖 |
| 符号 | API 名、`dlsym` | op 与 GetWorkspaceSize 都可解析 | `symbol_resolution_failed` | 比对库版本和导出符号 |
| 描述符 | tensor 摘要、创建结果 | 所有描述符非空 | `descriptor_creation_failed` | 检查 dtype/device/shape/stride |
| GetWorkspaceSize | 返回码、recent error、size、executor | 返回成功且 executor 可用 | `getworkspace_failed` | 固定参数和 tiling 日志 |
| Workspace | size、分配设备、生命周期 | 零值合法或非零分配成功 | `workspace_allocation_failed` | 区分 OOM、大小和释放 |
| 提交 | stream、返回码、recent error | ACLNN 提交返回成功 | `submit_failed` | 不跳过紧邻同步 |
| 同步 | 目标 stream/device 同步结果 | 同步成功 | `async_device_error` | 收集设备首错和 plog |
| 读取 | 输出元数据、拷贝参数 | 结果可安全访问 | `result_copy_failed` | 检查容量、方向、生命周期 |

## 库与符号证据

### 搜索路径

按真实实现记录顺序，不先手工重排：

```text
ASCEND_CUSTOM_OPP_PATH entries
  -> <entry>/op_api/lib/

ASCEND_OPP_PATH
  -> vendors/config.ini
  -> load_priority vendor list
  -> <vendor>/op_api/lib/
```

每个条目记录：

| Item | Value |
|---|---|
| 原始环境是否设置 | yes / no |
| 脱敏后的解析路径 |  |
| 目录是否存在 |  |
| 目标库是否存在 |  |
| 动态依赖是否完整 |  |
| 是否实际命中 |  |

文件存在、依赖可解析、库成功加载、符号成功解析是四个不同判据。

### 符号对

两段式 ACLNN 接口至少核对：

```text
aclnn<Op>GetWorkspaceSize
aclnn<Op>
```

只解析到其中一个时仍属于符号阶段。记录精确大小写、命中库和 `dlerror()`。

## 输入与描述符

为每个 tensor 填写：

| Tensor | Shape | Dtype | Device | Stride/contiguous | Descriptor |
|---|---|---|---|---|---|
| input 0 |  |  |  |  | non-null / null / unknown |
| input 1 |  |  |  |  |  |
| output 0 |  |  |  |  |  |

同时回答：

- 所有输入和输出是否位于同一目标 device？
- dtype 映射是否覆盖当前类型，还是落入默认分支？
- `.contiguous()` 是否改变了 tensor 对象和生命周期？
- shape/stride 数组在 `aclCreateTensor` 返回后是否可安全释放，由接口契约如何规定？
- 描述符创建失败是否在进入 GetWorkspaceSize 前被截获？

静态看不到检查时写 `preflight_gap`，不要自动写 `descriptor_creation_failed`。

## Workspace 与生命周期

| Object | Created at | Used until | Released at | Stream-aware | Status |
|---|---|---|---|---|---|
| input tensor |  |  |  |  |  |
| aclTensor descriptors |  |  |  |  |  |
| executor | GetWorkspaceSize | submit |  |  |  |
| workspace storage | after size | submit/execute |  |  |  |
| converted params |  |  |  |  |  |
| output tensor |  | result consumer |  |  |  |

需要特别区分：

- workspace 字节数和 tensor 元素数。
- `workspace_size == 0` 与分配失败。
- 提交返回时资源是否可释放，与设备实际完成时是否可释放。
- 框架 allocator 的 stream 语义是否有可见证据。

## Device 与 Stream

建立对照表：

| Layer | Device | Stream/context | Evidence |
|---|---|---|---|
| Python current device |  |  |  |
| input tensor |  | n/a |  |
| output tensor |  | n/a |  |
| C++ current device |  |  |  |
| ACLNN submit |  |  |  |
| synchronize |  |  |  |
| result copy |  |  |  |

判定：

- 所有行一致且有运行证据：device/stream 通过。
- 源码固定 device，但运行环境未知：`device_stream_needs_confirmation`。
- 实际 tensor 与 submit stream device 不一致并复现：`runtime_confirmed`。
- 同步了其他 stream：同步结果不能关闭目标阶段。

## 异步错误协议

### Python/cpp_extension

```python
torch.npu.synchronize()
output = custom_op(input)
torch.npu.synchronize()
```

### 直接 ACL

```cpp
ret = aclnnOp(workspace, workspaceSize, executor, stream);
// 先记录 ret 和 recent error
ret = aclrtSynchronizeStream(stream);
// 再记录同步 ret 和设备日志
```

### 结果解释

| Call | Sync | Copy/read | 结论 |
|---|---|---|---|
| fail | NOT_RUN | NOT_RUN | submit 或更早阶段 |
| success | fail | NOT_RUN | `async_device_error` |
| success | success | fail | `result_copy_failed` |
| success | success | success | 运行链路通过，数值问题另行处理 |
| success | missing | fail later | `blocked_missing_sync` |

提交成功不能替代同步成功；同步成功也不能替代精度验证。

## 仓库 dry-run

### `optimized_transducer`

静态映射：

| Stage | Repository evidence | Dry-run result |
|---|---|---|
| descriptor | pybind `toAclTensor` 创建描述符 | 返回值和 dtype 默认分支需要运行/预检确认 |
| GetWorkspaceSize | wrapper 直接调用两段式第一阶段 | 通用异常丢失返回码/recent error |
| workspace | PyTorch NPU allocator 分配字节 tensor | 实际分配 `NOT_RUN` |
| submit | wrapper 获取 NPU stream 后调用 ACLNN | 固定 device `0` 需要多设备确认 |
| synchronize | Python 与直接 ACL 样例均有显式同步 | 实际同步 `NOT_RUN` |
| read | 直接 ACL 样例同步后 device-to-host copy | 实际 copy `NOT_RUN` |

推荐分类：

```text
preflight_gap
diagnostic_context_loss
device_stream_needs_confirmation
NOT_RUN
```

### `unique_v3`

静态映射：

| Stage | Repository evidence | Dry-run result |
|---|---|---|
| library search | custom path 与 vendor priority | 搜索算法可见，实际命中 `NOT_RUN` |
| symbol | helper 解析 op 与 GetWorkspaceSize | 有空指针检查和符号错误文本 |
| GetWorkspaceSize | macro 调用并附 recent error | 实际返回 `NOT_RUN` |
| workspace | 非零时创建 NPU byte tensor | 实际分配 `NOT_RUN` |
| submit | `OpCommand` custom handler | 实际提交 `NOT_RUN` |
| synchronize | Python 测试调用前后同步 | 实际同步 `NOT_RUN` |

这里能确认的是诊断可观察性，不是运行成功。
