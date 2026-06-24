---
name: electric-inspection-shape-boundary
description: 当为 elec-ops-inspection 的 CANN、Ascend C、NPU 算子分析或审查 shape、rank、空张量、动态维度、tile/block 切换、尾块、对齐、packed 输入关系、workspace 派生量和整数溢出边界时使用；从 README、OpDef/infer、Host Tiling、Kernel 和 wrapper 推导最小边界用例与风险证据，不用于替代接口契约审查、完整测试计划或无证据声称运行缺陷。
---

# Electric Inspection Shape Boundary

使用本 skill 从算子源码推导 Shape 边界，而不是凭经验枚举几个尺寸。它关注逻辑 Shape 如何变成 Tiling 派生量、对齐长度、核数、循环次数和实际内存访问范围，并输出 reviewer 可以复核的公式、切换点和最小用例。

## 单一职责

本 skill 回答：

- 哪些 rank、维度和维度间关系是合法的？
- 哪些公式存在零除、上溢、下溢或截断风险？
- 哪些输入会切换 tile、block、tail、对齐或 Kernel 分支？
- 每个边界应测试前一项、边界值还是后一项，依据是什么？
- 缺少 NPU 时，哪些结论只是静态风险而不是已确认缺陷？

本 skill 不负责：

- 全量核对参数名称、顺序、dtype 和默认值；使用接口契约 Skill。
- 编写完整功能、精度、性能和回归测试计划。
- 根据 README 单方面判定源码错误，或根据静态分析声称 NPU 已崩溃。
- 修改公开 Shape 语义、自动生成文件或 Kernel 实现，除非用户明确要求修复。

## 输入要求

按实际存在情况读取：

- 根目录和目标算子 `README.md`、使用说明。
- OpDef、JSON、infer shape/dtype。
- Host Tiling 中的 Shape 读取、公式、workspace 和 `SetBlockDim`。
- TilingData 定义与 Kernel 的 `Init`、循环、分支、DataCopy 和 GM/UB 偏移。
- Python/aclnn wrapper 的输出分配和现有 examples。
- 当前 PR diff、失败日志和运行环境；没有 NPU 时明确标记 `dry_run`。

## 核心概念

区分以下量，禁止混用：

| 类型 | 示例 | 含义 |
|---|---|---|
| 逻辑 Shape | `[N]`、`[P, V]`、`[B, Umax]` | 用户和接口看到的维度 |
| 存储 Shape | `GetStorageShape()` | Host 实际读取的维度 |
| 有效长度 | `uniqueCnt`、每个样本的 `T/U` | 输出或 packed 数据中真正有效的区间 |
| 展平长度 | `totalLength`、`totalPositions` | 多维 Shape 乘积或 packed 总点位 |
| 对齐长度 | `ceil_div(N, A) * A` | 为 DataCopy、UB 或 workspace 扩展后的长度 |
| 调度量 | `tileNum`、`blockNum`、`usedCoreNum` | 决定 Kernel 启动和任务分配 |

## 工作流

### Step 1：建立 Shape 证据链

从每一层提取 Shape 事实：

1. README/wrapper：用户被告知的 rank、维度和有效区间。
2. OpDef/JSON/infer：框架接受和推导的 Shape。
3. Host Tiling：实际读取哪些维度、如何展平和派生调度量。
4. Kernel：如何消费 TilingData、循环次数、尾块和偏移。
5. Example：仓库当前真正构造过哪些 Shape。

每条事实附路径和行号。若各层冲突，只记录 `contract_conflict`，不擅自指定真源。

输出：Shape 证据链表。

### Step 2：写出派生公式

使用 [Shape 边界矩阵](references/shape-boundary-matrix.md) 记录：

- 输入维度和关系约束。
- `ceil_div`、乘积、取模、整除、对齐和 workspace 公式。
- 每个变量的 C++ 类型、单位和可能范围。
- 每个除数为何非零，每个乘积是否可能超过目标类型。
- Host 计算口径是否与 Kernel 偏移和循环一致。

不能只抄变量名；应把公式还原成可人工计算的表达式。

输出：公式表与前置条件。

### Step 3：推导切换点

为每个会改变执行路径的表达式寻找切换点：

- rank 或维度从非法到合法。
- 零维、最小合法值和空张量。
- `tileLength * k - 1`、`tileLength * k`、`tileLength * k + 1`。
- `tailLength == 0` 与非零尾块。
- `tileNum < coreNum`、`== coreNum`、`> coreNum`。
- 短块/长块数量发生变化的位置。
- `vocabSize` 小于、等于和大于单次 UB tile 容量的位置。
- `B` 小于、等于和大于可用核数的位置。
- 维度乘积接近 `uint32_t`、`int32_t` 或 `size_t` 边界的位置。

只选择能触发不同公式、分支或内存布局的点，避免无依据的“大中小”枚举。

输出：边界点、前/中/后用例和对应代码路径。

### Step 4：检查关系型 Shape

多个输入之间存在关系时，单独建立约束：

- batch 维是否一致。
- lengths 张量长度是否等于 batch。
- 每个 length 是否落在对应维度范围内。
- packed 第一维是否等于逐样本有效面积之和。
- 输出分配 Shape 是否覆盖 Kernel 写入范围。
- 动态有效长度是否被调用方正确裁剪。

为每条关系同时设计：

- 最小合法样例。
- 恰好违反一项关系的最小非法样例。
- 仓库是否有明确拒绝路径；没有时标记 `unguarded_static_risk`。

输出：关系约束矩阵。

### Step 5：分级静态结论

只使用以下结论：

- `guarded`：源码在进入危险公式或 Kernel 前显式拒绝。
- `derived_valid`：前置条件成立时，公式和访问范围静态闭合。
- `contract_conflict`：文档、注册、Tiling、Kernel 或 wrapper 的 Shape 定义互斥。
- `unguarded_static_risk`：可见路径缺少保护，危险表达式可由某个输入触发；尚未运行验证。
- `runtime_confirmed`：实际命令触发了预期错误或缺陷，并保留环境与日志。
- `not_covered`：仓库材料不足，无法判断。

静态除零、越界或溢出路径不能直接写成“已发生崩溃”；除非有运行证据，否则使用 `unguarded_static_risk`。

### Step 6：生成最小边界用例

使用稳定 ID：

- `RANK-*`：rank 和维度数量。
- `ZERO-*`：零维与空张量。
- `TILE-*`：tile 和尾块。
- `CORE-*`：单核/多核与负载分配。
- `ALIGN-*`：对齐和 DataCopy。
- `REL-*`：多输入 Shape 关系。
- `RANGE-*`：整数类型与乘积范围。

每个用例记录输入、公式代入值、预期进入的代码路径、预期状态和证据。没有 NPU 时，运行状态必须为 `NOT_RUN`。

输出：最小边界用例集，不扩展成完整测试计划。

### Step 7：整理 PR-ready 报告

使用 [Shape 边界报告模板](references/shape-boundary-report-template.md) 输出：

- Shape 证据链。
- 公式和前置条件。
- 切换点与最小用例。
- 受保护、未保护、冲突和未覆盖项。
- 实际执行命令、dry-run 结果和 NPU-only 待验证项。

## 仓库内算子观察点

### `unique_v3`

- README 描述一维输入，但 Host Tiling 当前对所有存储维度求乘积；先记录 rank 语义差异，不直接判断支持多维。
- `tileLength`、`tileNum`、`tailLength`、`blockNum` 和 `shortBlockTileNum` 构成连续派生链。
- 空输入、`N=1`、`8191/8192/8193`、`k*8192` 和跨 AIV 核数切换是优先边界。
- `totalLength` 使用 `uint32_t` 累乘，需检查大 Shape 乘积的表示范围。
- output/inverse 分配 Shape 与输入相同，但真正有效区间由 `uniqueCnt` 决定。

### `optimized_transducer`

- logits 预期为 `[P, V]`，其中 `V > 0`，`P` 应与逐样本 `T*(U+1)` 之和一致。
- targets、logit_lengths、target_lengths 的 batch 维需要形成闭环。
- Host 直接读取 targets 的第 1 维，需检查 rank 前置条件是否有显式保护。
- `totalLength / vocabSize`、`batchSize / usedCoreNum` 和 vocab tile 循环均要求非零前置条件。
- 优先覆盖 `V=0/1`、`B=0/1`、不一致 batch、packed 面积不一致、长度最小值和 vocab 对齐前后。

这些观察点用于验证分析方法，不是未经运行确认的缺陷清单。

## 错误处理

- Shape 只在 README 出现：标记 `not_covered`，继续寻找注册和 Tiling 证据。
- 动态 Shape 在编译期未知：区分编译期约束和运行期值，列出需要 NPU/框架验证的用例。
- 平台核数或 UB 大小未知：保留符号变量，给出切换公式，不编造具体硬件值。
- 公式单位不明确：停止数值代入，先标记元素数、字节数、block 数或 tile 数。
- 自动生成文件与源文件冲突：记录生成链缺失，不直接修改生成结果。
- 分析发现疑似缺陷：先产出最小输入和代码路径；用户要求修复后再修改源码。

## 验证

至少执行：

```bash
git diff --check
rg -n "GetDim|GetShapeSize|tileNum|blockNum|tailLength|SetBlockDim|Workspace|DataCopy" <operator-dir>
rg -n "(A[K]IA|BEGIN [A-Z ]*PRIVATE K[E]Y|GITCODE[_]TOKEN|Authorizatio[n]:|passwor[d]\\s*=|t[o]ken\\s*=|s[e]cret\\s*=)" <changed-paths>
```

对一个现有算子完成手工公式代入，证明 Skill 能产出具体边界和非空结论。有 NPU 时再运行最小合法与非法用例；没有时标记 `dry_run`。

## 输出格式

最终返回：

1. Shape 证据链与契约冲突。
2. 派生公式、类型、单位和前置条件。
3. 执行路径切换点。
4. 最小边界用例集。
5. `guarded / unguarded_static_risk / runtime_confirmed / not_covered` 结论。
6. 已运行验证、未验证项和 PR 描述片段。

## 参考资料

- [Shape 边界矩阵](references/shape-boundary-matrix.md)
- [Shape 边界报告模板](references/shape-boundary-report-template.md)
