---
name: electric-inspection-build-install-triage
description: 当 elec-ops-inspection 的 CANN、Ascend C、NPU 算子在 CMake 配置、build.sh 编译、binary/package 目标、.run 打包安装、vendor 目录部署、cpp_extension/pybind 构建、wheel 或 editable install、Python import 前置阶段失败时使用；通过首个失败命令、预期产物和依赖链定位构建安装根因，不用于算子成功加载后的 ACL/NPU 运行时、精度或性能诊断。
---

# Electric Inspection Build and Install Triage

使用本 skill 定位 `elec-ops-inspection` 算子从源码到“可被调用方加载”之间的构建与安装故障。诊断以第一个失败阶段为中心，不用后续的 `ModuleNotFoundError`、动态库报错或测试失败掩盖更早的配置、打包或安装问题。

## 单一职责

本 skill 处理：

- 找不到或用错 `build.sh`、`CMakeLists.txt`、preset、target、工作目录。
- CANN、编译器、Python、CMake、PyTorch/torch_npu 路径发现失败。
- Host、Kernel、binary、package 或 `.run` 产物缺失。
- `.run` 安装后 vendor 目录、头文件、动态库或配置文件缺失。
- cpp_extension、pybind、wheel、editable install 构建失败。
- Python 扩展因安装路径、RPATH 或依赖库不可见而无法 import。

本 skill 不处理：

- Python 扩展已成功 import 后，ACLNN workspace、设备执行、流同步或 NPU 错误。
- output、loss、grad、inverse、counts 的精度差异。
- benchmark、性能回归或显存优化结论。
- 未经用户授权执行系统级安装、删除 vendor 目录或修改 shell 启动文件。

## 输入要求

开始前收集：

- 精确命令、执行目录和退出码。
- 完整日志中第一个 error，以及其前后必要上下文。
- `git rev-parse HEAD` 和目标算子路径。
- 操作系统/架构、CANN、CMake、编译器、Python、PyTorch、torch_npu 版本。
- 相关环境变量的“是否设置与解析后路径”，不要收集令牌或凭证。
- 预期产物和实际产物列表。
- 是否允许写入 CANN vendor 目录、Python 环境或系统路径。

如果只有最后一行错误或截图，先要求文本日志和原始命令；不要猜根因。

## 阶段模型

| 阶段 | 成功证据 | 典型失败 |
|---|---|---|
| 0. 入口 | 命令引用仓库内存在的文件，工作目录明确 | 脚本/目录不存在、命令来自其他仓库 |
| 1. 环境发现 | CANN、编译器、Python、CMake 路径可解析 | 环境变量缺失、版本不匹配、工具不在 PATH |
| 2. CMake 配置 | 生成 `CMakeCache.txt` 和构建系统 | preset、package、include、compiler 查找失败 |
| 3. 编译 | 目标库、Kernel 或 Host 产物生成 | 编译、模板、头文件、链接失败 |
| 4. 打包 | `.run`、shared library 或 wheel 生成 | target 不存在、CPack/makeself、产物命名不符 |
| 5. 安装 | vendor/Python 环境出现预期文件 | 权限、路径、vendor 名称、覆盖策略失败 |
| 6. 加载验收 | Python import 或动态库依赖解析成功 | 模块不可见、RPATH/LD_LIBRARY_PATH、缺失 `.so` |

只有上一阶段有证据通过，才进入下一阶段。算子调用已经开始后，移交 Runtime Triage，不继续归因于“安装问题”。

## 工作流

### Step 1：冻结原始证据

记录：

- 原始命令与当前目录。
- 第一个失败命令和退出码。
- 第一个 error，而非日志末尾的连锁错误。
- 已生成的最后一个可信产物。
- 用户已尝试的命令及结果。

不要一开始就清理 `build_out/`；CMakeCache、链接命令和中间产物可能是关键证据。

输出：最小复现头部。

### Step 2：验证仓库内入口

逐个确认命令中的相对路径确实存在：

```text
build.sh
CMakeLists.txt
CMakePresets.json
setup.py
pyproject.toml
目标 example/test
```

同时确认命令应从仓库根目录、算子目录还是扩展目录执行。若文档引用外部 monorepo 的脚本或旧目录，分类为 `entrypoint_not_in_repo`，不要尝试虚构替代参数。

输出：入口路径矩阵和建议工作目录。

### Step 3：建立环境解析表

使用 [失败阶段矩阵](references/build-install-stage-matrix.md) 检查：

- `ASCEND_HOME_PATH`、`ASCEND_OPP_PATH`、`ASCEND_CUSTOM_OPP_PATH` 的用途与实际路径。
- CMake 实际读取的是环境变量、cache 变量还是 preset 硬编码值。
- Python、pip、torch、torch_npu 是否来自同一环境。
- 主机架构与目标包架构。
- CMake 和编译器是否满足仓库当前脚本，而不是只满足 README。

只报告变量是否存在、路径是否存在和版本；敏感环境变量的值不得进入日志。

输出：环境解析表和冲突项。

### Step 4：定位配置或编译失败

配置失败时检查：

- 使用了哪个 preset 和 cache 变量。
- `CMakeCache.txt` 中的 CANN、编译器、Python、Torch、vendor 路径。
- configure 命令是否被脚本重写。

编译失败时检查：

- 失败 target 和第一条 compiler/linker error。
- 缺失的是源码、生成文件、头文件、库还是符号。
- Host、Kernel、framework 和 Python extension 哪一层正在编译。
- 并行构建是否让首错被后续错误淹没；必要时建议单线程重跑失败 target。

输出：失败层和最小重跑命令。

### Step 5：验证打包产物

不要只检查“命令返回 0”。记录：

- 实际 target：`binary`、`package`、`install`、wheel 等。
- 预期文件名模式和实际文件名。
- 文件架构、大小和生成时间。
- `.run` 内 vendor 名称与构建参数是否一致。
- 文档命令是否引用仓库实际会生成的产物。

找不到产物时先检查 target 和输出目录，再检查通配符。不要把旧构建目录中的历史文件当作本次结果。

输出：产物清单和 `package_missing`/`artifact_name_mismatch` 结论。

### Step 6：验证安装树

安装前必须获得用户对写入目标环境的授权。安装后只读验证：

- 安装目标路径。
- vendor 名称。
- op_api 头文件与库。
- op_proto、op_impl、配置和 Kernel 产物。
- 文件时间与本次包是否对应。
- 安装脚本是否要求 `ASCEND_OPP_PATH` 或自定义安装路径。

不要自动删除整个 vendor 目录。旧版本冲突时先列出差异，再请求用户选择覆盖、并存或卸载。

输出：安装树验收和缺失组件。

### Step 7：诊断 Python 扩展

区分：

1. `setup.py`/CMake 配置失败。
2. C++ extension 编译或链接失败。
3. wheel/editable install 失败。
4. 包已安装但模块路径不可见。
5. 模块文件存在但动态依赖/RPATH 解析失败。

检查：

- `python -m pip` 与运行测试的 Python 是否相同。
- 包名、扩展模块名和 `import` 名是否一致。
- `torch.utils.cmake_prefix_path`、`NPU_PATH`、CANN include/lib 和 vendor 路径。
- exact-version `install_requires` 是否触发意外依赖解析。
- `.so` 的动态依赖是否都能解析。

当 import 已成功且错误发生在 ACLNN 调用后，停止并移交 Runtime Triage。

输出：Python 扩展失败层和验证命令。

### Step 8：给出最小修复与回归路径

优先修复第一个失败阶段。建议必须包含：

- 要修改的命令、路径、配置或文件。
- 为什么它是首因证据。
- 修复后只重跑哪一步。
- 该步骤成功时应出现什么产物。
- 后续阶段哪些仍为 `NOT_RUN`。

不要用全量重装掩盖路径错误，也不要在未确认依赖版本前随意升级 CANN、PyTorch 或 torch_npu。

### Step 9：整理 PR-ready 报告

使用 [构建安装诊断报告模板](references/build-install-report-template.md)，输出：

- 阶段和首个失败点。
- 命令、工作目录与环境摘要。
- 预期/实际产物。
- 根因证据与最小修复。
- 已验证、未验证和需要授权的动作。

## 仓库内算子观察点

### `unique_v3`

- 独立 `build.sh` 与 `CMakePresets.json` 位于算子目录；工作目录应明确。
- `CMakePresets.json` 同时包含 CANN 路径、SoC、vendor 和 package 配置，需确认实际 cache 值。
- `.run` 安装后应核对 `vendors/customize` 与 op_api/op_impl 产物。
- cpp_extension 的 package 名、扩展模块名和 import 名不同，需要分别验证。
- 文档和辅助脚本中的测试路径必须与当前仓库树一致。

### `optimized_transducer`

- 当前算子目录没有独立 `build.sh`；文档中的 build 命令可能依赖外部算子工程，必须标明来源。
- pybind 使用 `setup.py -> CMake -> shared library` 链路，并依赖 Torch、torch_npu、CANN 和 vendor op_api。
- 文档中的 vendor、绝对路径和 pybind 工作目录需要与当前仓库结构逐项核对。
- `pip install -e . --no-build-isolation` 不代表跳过依赖解析；需记录实际 Python 环境和安装行为。

这些观察点用于 dry-run，不自动修改现有文档或构建脚本。

## 分类

- `entrypoint_not_in_repo`
- `environment_unresolved`
- `configure_failed`
- `compile_failed`
- `package_missing`
- `artifact_name_mismatch`
- `install_tree_incomplete`
- `extension_build_failed`
- `import_visibility_failed`
- `verification_signal_invalid`
- `blocked_missing_log`
- `runtime_handoff`

每个结论必须绑定命令、路径、日志或产物证据。

## 错误处理

- 日志不完整：标记 `blocked_missing_log`，只提供收集命令。
- 无 Linux/CANN 环境：执行路径、配置和产物链 dry-run，不声称编译成功。
- 需要安装依赖或写系统目录：先请求用户授权。
- 需要删除 build/vendor/Python 环境：列出目标路径和影响，未经授权不执行。
- 文档命令来自外部仓库：标记依赖来源，不在当前仓库伪造脚本。
- 同时出现多个错误：只修首个失败阶段，后续错误保留为待复验。

## 验证

至少执行：

```bash
git diff --check
rg -n "build\\.sh|CMake|\\.run|vendor|ASCEND|setup\\.py|pip|wheel|LD_LIBRARY_PATH" <operator-dir>
rg -n "(A[K]IA|BEGIN [A-Z ]*PRIVATE K[E]Y|GITCODE[_]TOKEN|Authorizatio[n]:|passwor[d]\\s*=|t[o]ken\\s*=|s[e]cret\\s*=)" <changed-paths>
```

对两个现有算子做路径与产物 dry-run，至少找出一个可定位的首个断点。实际构建、安装和 import 未执行时分别标记 `NOT_RUN`。

## 输出格式

最终返回：

1. 失败阶段和首个错误。
2. 命令、工作目录和环境解析。
3. 预期/实际产物。
4. 根因证据与最小修复。
5. 重跑命令和成功判据。
6. 已验证、未验证、需授权动作和 runtime 移交项。

## 参考资料

- [构建安装失败阶段矩阵](references/build-install-stage-matrix.md)
- [构建安装诊断报告模板](references/build-install-report-template.md)
