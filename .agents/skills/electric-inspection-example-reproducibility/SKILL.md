---
name: electric-inspection-example-reproducibility
description: 当为 elec-ops-inspection 新增、审查或修复算子 examples、README 调用示例、最小复现命令、合成输入数据、CPU/NPU 对照输出和无 NPU 降级说明时使用；适用于 CANN/Ascend C 电力巡检算子贡献的示例可复现性检查，不用于 benchmark 造数或 PR 通用预检。
---

# Electric Inspection Example Reproducibility

使用本 skill 检查 `cann/elec-ops-inspection` 算子示例是否能被 reviewer 和新贡献者复现。它关注“能不能按文档跑起来、输出是否可对照、缺少 NPU 时如何说明”，不负责性能报告或泛化 PR 模板。

## 适用范围

- 新增或修改 `examples/`、`pybind/test.py`、`README.md` 调用示例、`doc/使用说明.md`。
- 为 `optimized_transducer`、`unique_v3` 或后续电力巡检算子补齐最小运行路径。
- 审查示例是否依赖私有数据、本地绝对路径、隐藏环境变量或不可复现输出。
- 给 PR 描述整理“示例验证”章节，说明运行命令、期望输出、缺失环境和降级说明。

不要使用本 skill 声称性能提升、硬件兼容性或现场业务收益；这些必须由专门 benchmark 或端到端验证支撑。

## 工作流

1. 读取目标文件：
   - 根目录 `README.md`，确认仓库定位和算子列表。
   - 算子目录的 `README.md`、`doc/使用说明.md`、`examples/`、`pybind/`。
   - 构建入口，例如 `build.sh`、`CMakeLists.txt`、`setup.py`、安装脚本。
2. 建立示例路径：
   - 写清从干净 checkout 到运行示例的最短命令序列。
   - 区分构建、安装、运行、验证四个阶段。
   - 明确是否需要 NPU、CANN、torch_npu、torchaudio 或自定义算子包。
3. 检查输入与输出：
   - 使用 `references/example-reproducibility-checklist.md` 审查输入 shape、dtype、合成数据、随机种子和期望输出。
   - 示例输出要能和 CPU/reference 或已知不变量对照。
   - 无法在当前环境运行时，保留 dry-run 检查项，不写“已通过”。
4. 整理 PR 说明：
   - 列出命令、预期输出、未运行原因和需要 reviewer 在 NPU 环境确认的步骤。
   - 如示例依赖私有数据，改为合成数据或公开数据；无法替代时标记 blocked。

## 检查重点

- 命令可复制：不包含本地绝对路径、个人目录、临时文件名或不可见环境变量。
- 数据可获得：优先使用脚本内合成数据；现场数据只作为可选输入，不作为默认运行条件。
- 输出可判定：提供 shape、dtype、误差阈值、唯一值数量、loss 范围或布尔检查。
- 依赖可解释：缺少 NPU 或 CANN 时说明哪些步骤只能 dry-run。
- 场景不夸大：只说明示例覆盖的算子输入，不把小样例写成真实巡检系统验证。

## 输出格式

```markdown
## Example Reproducibility

- Operator:
- Example path:
- Required environment:
- Commands:
- Input data:
- Expected output:
- Reference check:
- Dry-run result:
- NPU-only validation:
- Missing items:
```

## 参考资料

- [示例可复现性检查清单](references/example-reproducibility-checklist.md)
