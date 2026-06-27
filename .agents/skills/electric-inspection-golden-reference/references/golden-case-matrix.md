# Golden 用例矩阵

读取时机：为 `elec-ops-inspection` 算子选择 Golden 输入、比较方法和边界覆盖时读取。

本文件用于把“测几个随机输入”转换为可审查的用例矩阵。它补充 `SKILL.md` 的 Step 3 和 Step 4，不替代目标算子的真实接口约束。

## 通用字段

每个用例至少记录：

| 字段 | 内容 |
|---|---|
| `case_id` | 稳定、可在日志和 PR 中引用的编号 |
| semantic source | 代码、文档、上游 API 或数学定义 |
| dtype / shape / layout | 完整输入规格 |
| attributes | 所有影响语义的属性 |
| data pattern | 固定值、结构化输入或随机分布 |
| seed | 随机用例的固定种子；非随机写 `N/A` |
| Golden source | 参考 API、独立实现、手算值或不变量 |
| comparison | exact、`atol/rtol` 或 invariant |
| expected status | 支持、拒绝或待确认 |
| actual status | `PASS / FAIL / BLOCKED / NOT_RUN` |

## 基础因子

### 规模与对齐

- 最小有效输入。
- 单元素或单样本。
- 小于一个 tile/block 的输入。
- 刚好等于 tile/block 边界的输入。
- 边界前一位和后一位。
- 多 tile、多 block 和尾块输入。
- 当前代码或文档声明的最大规模；无法运行时只登记，不声称通过。

### 数据分布

- 全部相同。
- 全部不同。
- 少量高频值与长尾值。
- 重复值跨 tile/block 边界。
- 已排序、逆序和随机顺序。
- 零、正值和负值。
- 极值、无穷和 NaN，仅在语义允许时覆盖。

### 属性组合

- 默认属性。
- 每个布尔属性独立开启。
- 关键布尔属性的笛卡尔组合。
- 数值属性的默认值、有效边界和非法值。
- 可选输出关闭时，确认调用方不会读取未定义内容。

## `unique_v3` 建议矩阵

| 类别 | 输入示例 | 主要判定 |
|---|---|---|
| 最小输入 | `[7]` | `uniqueCnt == 1`；有效 output 为 `[7]` |
| 全相同 | `[3, 3, 3, 3]` | 唯一值一个；counts 为输入长度 |
| 全不同 | `[4, 1, 3, 2]` | 有效 output 与参考排序唯一值一致 |
| 结构化重复 | `[2, 1, 2, 3, 1]` | output、inverse、counts 同时对照 |
| 尾块 | 长度为 tile 边界前后 | 有效长度、不变量和尾部处理 |
| 属性组合 | inverse/counts 四种组合 | 只验证已启用输出 |
| dtype | 每个已确认支持类型 | 精确比较；记录转换限制 |
| 特殊值 | Inf/NaN/大整数 | 先确认语义；未知时 `BLOCKED` |

必须增加的不变量：

- `uniqueCnt` 等于有效唯一值数量。
- 已启用 inverse 时，按 inverse 重建输入。
- 已启用 counts 时，有效 counts 之和等于输入长度。
- output 有效区间满足已确认的排序语义。

## `optimized_transducer` 建议矩阵

| 类别 | 输入设计 | 主要判定 |
|---|---|---|
| 最小格点 | 小 `B/T/U/V` | loss、grad shape 和有限值 |
| 多样本 | 不同 `T/U` | packed 面积与 lengths 一致 |
| blank | 首位、末位、已确认的负索引 | loss 与 grad 对照 |
| clamp | 禁用、较小正值、较大正值 | grad 截断语义 |
| fused softmax | true/false（若支持） | 与匹配参考路径对照 |
| dtype | 仅覆盖源码确认支持的类型 | loss/grad 容差比较 |
| 极端 logits | 大正值、大负值、相同值 | 有限性和数值稳定性 |
| 非法输入 | 面积不一致、长度越界 | 预期拒绝或明确错误 |

比较时同时记录：

- loss 最大绝对误差。
- grad 最大绝对误差和最大相对误差。
- 误差最大位置及对应输入索引。
- NaN/Inf 数量。
- 参考 API、版本、reduction 和 softmax 设置。

## 容差记录

浮点比较必须保留：

| 字段 | 示例 |
|---|---|
| `atol` | `1e-4` |
| `rtol` | `1e-4` |
| rationale | 上游测试惯例、误差分析或仓库既有标准 |
| max abs error | 实际测量值 |
| max rel error | 实际测量值 |
| failing index | 首个或最大误差位置 |

禁止只写“误差很小”“基本一致”或在看到结果后无依据地调整阈值。
