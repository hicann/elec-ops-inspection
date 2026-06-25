# 构建安装失败阶段矩阵

读取时机：构建、打包、安装或 Python 扩展加载失败时读取。

本文件用于识别第一个失败阶段、收集最小证据并选择重跑命令。核心流程和远端操作边界以 `SKILL.md` 为准。

## 导航

- [阶段矩阵](#阶段矩阵)
- [环境解析](#环境解析)
- [产物验收](#产物验收)
- [Python 扩展](#python-扩展)
- [仓库 dry-run 提示](#仓库-dry-run-提示)

## 阶段矩阵

| 阶段 | 输入 | 成功证据 | 首错证据 | 不要做 |
|---|---|---|---|---|
| 入口 | 命令、cwd、仓库树 | 文件和目录存在 | `not found`、路径不存在 | 猜测外部脚本参数 |
| 环境 | PATH、版本、CANN/Python 路径 | 工具可执行、路径存在 | 未设置、版本冲突 | 输出凭证或全量环境 |
| 配置 | preset/cache/源码 | `CMakeCache.txt` | configure 第一条 error | 直接删 build 证据 |
| 编译 | 构建系统、生成文件 | 目标库/对象存在 | compiler/linker 首错 | 只看最后一行 |
| 打包 | binary/install target | `.run`/wheel/shared lib | target/CPack/文件缺失 | 复用旧产物 |
| 安装 | 包与目标目录 | vendor/Python 文件齐全 | 权限/目录/覆盖失败 | 未授权删除 vendor |
| 加载 | 模块与动态依赖 | import/依赖解析成功 | module/so/RPATH 错误 | 混入 NPU 运行诊断 |
| 验收信号 | 状态码、产物、检查命令 | 信号与真实结果一致 | 失败后仍打印成功 | 只相信成功字符串 |

## 环境解析

| 项目 | 收集方式 | 判定 |
|---|---|---|
| 当前目录 | `pwd` | 是否与文档一致 |
| Commit | `git rev-parse HEAD` | 是否为待诊断版本 |
| 主机架构 | `uname -m` | 是否匹配包/编译器 |
| CMake | `cmake --version` | 与脚本和 preset 比较 |
| Python/pip | `python -c ...`、`python -m pip --version` | 是否同一环境 |
| Torch/NPU | import 后打印模块路径和版本 | 是否来自预期环境 |
| CANN | 解析后的安装路径和 version 文件 | 路径是否存在 |
| vendor | 构建参数、安装目录、扩展链接目录 | 名称是否一致 |

环境变量只记录“已设置/未设置”和脱敏路径。名字包含 token、password、secret、authorization 的变量不要输出值。

## 产物验收

### CANN 算子包

至少记录：

```text
build directory
binary target result
.run file path
package architecture
vendor name
op_api headers and libraries
op_proto files
op_impl/kernel/tiling files
```

### Python 扩展

至少记录：

```text
distribution/package name
extension module name
Python import name
wheel or editable install metadata
compiled .so path
dynamic dependencies
```

不要假设这三个名字相同。例如 distribution 可以是 `custom_ops`，底层 extension 可以是 `custom_ops_lib`，用户 import 仍可能是 `custom_ops`。

## Python 扩展

| 症状 | 优先检查 |
|---|---|
| `No module named torch_npu` | Python 环境、依赖是否已安装 |
| CMake 找不到 Torch | `Torch_DIR` 和当前 torch 模块路径 |
| 找不到 ACL/CANN 头文件 | CANN/NPU include 路径 |
| 找不到 `cust_opapi` | vendor 名称、安装树、link directories |
| wheel 生成但 import 失败 | 安装 Python、包名、extension `.so` |
| `.so` 存在但加载失败 | 动态依赖、RPATH、`LD_LIBRARY_PATH` |
| import 成功但 ACL 调用失败 | 移交 Runtime Triage |

## 仓库 dry-run 提示

### `unique_v3`

检查：

- `build.sh` 只存在于 `unique_v3/`，文档工作目录是否明确。
- `CMakePresets.json` 的 vendor、SoC、CANN 路径和文档是否一致。
- 文档中的每条测试命令是否指向存在的文件。
- `examples/cppExtension/build_and_run.sh` 引用的 `test/` 和测试文件是否存在。
- `.run` 名称来自 CPack 实际规则还是示例占位。

### `optimized_transducer`

检查：

- 当前仓库根目录和算子目录是否存在文档引用的 `build.sh`。
- `experimental/loss/optimized_transducer/pybind` 是否存在于当前树。
- 文档的 vendor 名称与 pybind CMake 的 `custom_nn` 路径是否一致。
- `ASCEND_HOME_PATH` fallback、Torch/NPU 路径和 exact dependencies 是否可解释。

路径不存在可静态确认为 `entrypoint_not_in_repo`。编译、链接和安装结果仍需目标环境验证。
