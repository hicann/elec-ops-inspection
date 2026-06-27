---
name: electric-inspection-golden-reference
description: 当为 elec-ops-inspection 算子设计、审查或修复 Golden 参考结果、CPU/reference 对照、测试向量、随机种子、精确与近似比较规则、loss/grad/output/inverse/counts 正确性判定时使用；适用于 CANN、Ascend C、NPU 电力巡检算子的可信 oracle 建立，不用于性能 benchmark、伪造实测结果或替代完整测试计划。
---

# Electric Inspection Golden Reference

使用本 skill 为 `cann/elec-ops-inspection` 算子建立可解释、可复现、独立于被测实现的 Golden 参考结果。目标是让贡献者和 reviewer 能回答三个问题：

1. 期望结果从哪里来？
2. 为什么这个参考结果可信？
3. 当前输出应按精确一致、容差比较还是不变量验证来判定？

## 适用范围

- 为新算子或现有算子补充 CPU/reference 实现和测试向量。
- 审查 `unique_v3` 的唯一值、有效长度、inverse 和 counts。
- 审查 `optimized_transducer` 的 loss、grad、blank、clamp 和长度约束。
- 为 PR 整理 Golden 来源、输入生成、比较规则、失败样例和未验证项。
- 在没有 NPU 的环境中先完成参考侧与静态一致性检查。

不要使用本 skill：

- 生成或背书没有运行证据的 NPU 结果。
- 用被测 Kernel 的等价改写充当“独立”参考实现。
- 用 Golden 正确性结果替代性能 benchmark、端到端业务验收或完整测试计划。
- 在语义、dtype、shape 或属性约束尚未确认时自行补全接口事实。

## 输入要求

开始前收集：

- 根目录 `README.md` 和目标算子 `README.md`、使用说明。
- 接口定义、Host Tiling、Kernel、Python/aclnn 包装和现有样例。
- 参考语义来源，例如 PyTorch/torchaudio API、明确的数学定义或独立朴素实现。
- 输入 dtype、shape、layout、属性、随机种子和特殊值策略。
- 当前测试命令、原始输出、失败日志和运行环境；缺少时标记 `NOT_RUN` 或 `BLOCKED`。

## 工作流

### Step 1：确认被测语义

1. 从代码和文档提取输入、输出、属性及有效区间。
2. 对照 op 定义、Tiling、Kernel 和调用包装，记录不一致项。
3. 为每个输出写出可判定语义，不只写“结果正确”。
4. 如果代码与文档对 dtype、shape、可选输出或属性含义冲突，先标记 `BLOCKED: contract conflict`，不要选择对自己有利的一侧继续测试。

输出：一份被测契约摘要和待确认冲突列表。

### Step 2：选择 Golden 来源

按以下优先级选择：

1. 公开且语义明确的上游参考 API。
2. 独立、简单、可人工审查的 CPU 实现。
3. 数学定义生成的已知小样例。
4. 不变量或变形关系，用于无法直接构造完整期望值的场景。

参考实现不得复用被测 Kernel 的排序、分块、Tiling 或中间缓冲逻辑。若参考 API 的版本会影响语义，记录库名和版本。

输出：Golden 来源、独立性说明和版本信息。

### Step 3：建立测试向量矩阵

读取 [Golden 用例矩阵](references/golden-case-matrix.md)，至少覆盖：

- 最小有效输入、典型输入和当前实现宣称的边界规模。
- 每个支持 dtype 与属性组合。
- 全相同、全不同、重复跨块、尾块、非对齐长度等结构化输入。
- 零、负值、极值、无穷和 NaN；仅在接口语义允许时加入。
- 随机输入必须固定种子，并记录生成算法和数据分布。

每个用例只声明已由代码、文档或维护者确认的支持范围。未知行为写入待确认项。

输出：带 `case_id` 的输入与预期判定矩阵。

### Step 4：定义比较规则

- 离散结果、索引和计数优先逐元素精确比较。
- 浮点 loss、grad 或近似算法同时记录 `atol`、`rtol`、最大绝对误差和最大相对误差。
- 容差必须来自参考 API 惯例、数值分析、仓库标准或已有可信测试，不能为了让失败样例通过而临时放宽。
- 动态有效长度输出只比较有效区间，并单独验证长度值。
- 除直接对照外，增加不变量检查，避免两个实现以同样方式出错。

输出：每个输出字段的比较方法、阈值依据和不变量。

### Step 5：执行并保留首个失败证据

1. 先在 CPU/reference 侧运行小样例，确认 Golden 本身可解释。
2. 环境允许时运行 NPU 输出并同步设备后比较。
3. 记录第一个失败 `case_id`、输入摘要、期望值、实际值、误差位置和命令。
4. 不覆盖原始失败结果；修复后新增一轮记录。
5. 无 NPU 时只报告参考侧结果和静态检查，不写“算子测试通过”。

状态统一使用：

- `PASS`：命令实际执行且满足比较规则。
- `FAIL`：命令实际执行但结果不满足规则。
- `BLOCKED`：契约、依赖、数据或环境缺失，无法形成有效比较。
- `NOT_RUN`：本轮未执行，且不推断结果。

输出：逐用例结果与首个失败证据。

### Step 6：整理 PR-ready 结论

使用 [Golden 验证报告模板](references/golden-report-template.md) 输出：

- Golden 来源与独立性。
- 用例范围与随机种子。
- 比较规则和阈值依据。
- 实际运行命令与结果。
- 失败样例、契约冲突、未覆盖风险和 NPU-only 待验证项。

## 仓库内算子检查点

### `unique_v3`

- 只比较 `output[:uniqueCnt]` 的有效区间。
- 验证 `output[inverse[i]] == input[i]`，前提是 inverse 输出已启用且语义已确认。
- 验证有效 `counts` 之和等于输入元素数，并与唯一值逐项对应。
- 分别覆盖 `flag_inverse`、`flag_counts` 的开启和关闭组合。
- 对整数、浮点、无穷、NaN 和超出可精确转换范围的值，不假设行为；先从代码和文档确认。

### `optimized_transducer`

- 明确参考 API、版本、`reduction` 和 `fused_log_softmax` 设置。
- 验证 packed logits 与 `logit_lengths`、`target_lengths` 的面积关系。
- 分别比较 loss 和 grad；梯度比较前确认 shape 还原方式一致。
- 覆盖不同 `blank`、`clamp` 和长度组合，并检查结果是否为有限值。
- 如果 op 定义、Tiling 校验、README 与 Python 包装声明的 dtype 不一致，先报告契约冲突。

## 错误处理

- 找不到独立参考：使用小规模数学样例和不变量，报告覆盖范围为 `PARTIAL`；未形成有效对照的 case 标记 `BLOCKED` 或 `NOT_RUN`。
- 参考 API 版本不明：标记 `BLOCKED`，先补版本信息。
- 两个参考来源不一致：保留双方输出，缩小到最小样例并请求确认语义。
- 只有截图或汇总表：要求原始命令、输入生成和机器可读结果；否则不能作为 Golden 证据。
- NPU 不可用：完成 reference dry-run 和静态契约检查，NPU 结果标记 `NOT_RUN`。
- 容差无依据：标记 `BLOCKED: tolerance rationale missing`。

## 输出格式

最终至少包含：

```markdown
## Golden Reference Validation

- Operator:
- Contract source:
- Golden source and version:
- Independence rationale:
- Case matrix:
- Seed and data generation:
- Comparison rules:
- Commands executed:
- Results:
- First failing case:
- Contract conflicts:
- NPU-only validation:
- Unverified items:
```

## 参考资料

- [Golden 用例矩阵](references/golden-case-matrix.md)
- [Golden 验证报告模板](references/golden-report-template.md)
