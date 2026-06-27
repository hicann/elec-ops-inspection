---
name: electric-inspection-tiling-review
description: 当新增、修改或审查 elec-ops-inspection 的 CANN、Ascend C、NPU 算子 Host Tiling、TilingData、tiling key、blockDim、核间任务分配、workspace 分区、GM 偏移和 Kernel 消费逻辑时使用；用于建立字段生命周期与字节级内存地图，发现未消费字段、重复常量、单位漂移和容量风险，不用于替代公开接口契约、Shape 边界分析、性能 benchmark 或无运行证据声称越界。
---

# Electric Inspection Tiling Review

使用本 skill 审查 `elec-ops-inspection` 算子的 Host Tiling 与 Kernel 调度闭环。核心不是确认字段“名字对得上”，而是证明每个字段、调度决策和 workspace 字节都在正确的单位下被计算、传递和消费。

## 单一职责

本 skill 回答：

- TilingData 每个字段由谁定义、在哪些成功路径赋值、是否序列化、由谁读取、最终影响什么行为？
- `SetBlockDim`、tiling key 和 Kernel 模板/分支是否形成可解释的调度闭环？
- Host 申请的 workspace 是否覆盖 Kernel 使用的最后一个字节？
- 元素数、字节数、float/int32 指针偏移、对齐数量是否被混用？
- 同一个 tile 大小、对齐值或核数是否存在多个真源，未来修改时会不会漂移？

本 skill 不负责：

- 核对公开输入输出、dtype 和属性语义；使用接口契约 Skill。
- 推导 Shape 的合法域和前/中/后边界；使用 Shape Boundary Skill。
- 证明性能收益或寻找最优 tile 参数。
- 仅凭静态代码宣布越界、数据破坏、死锁或运行失败。

## 输入要求

至少读取：

- Host Tiling 实现和 TilingData 定义。
- Kernel 入口、TilingData 读取、`Init` 和调度分支。
- workspace 申请公式、所有 `SetGlobalBuffer`、GM 指针转换和偏移。
- `SetBlockDim`、tiling key 设置、Kernel 模板参数或 key 分支。
- 与 tile、block、对齐、同步事件和平台信息相关的常量。
- 当前 PR diff；有运行失败时再读取命令、环境和日志。

不要把 README 中的架构描述当作内存布局证据。workspace 结论必须回到 Host 公式和 Kernel 指针范围。

## 核心概念

### 字段生命周期

一个有效 Tiling 字段通常经历：

```text
定义 -> Host 计算 -> Host 赋值 -> 序列化/传递 -> Kernel 读取 -> 影响分支、循环、偏移或容量
```

字段存在于 struct 并不代表它参与运行。反过来，Kernel 硬编码的常量也可能绕过 TilingData，形成第二真源。

### workspace 内存地图

每个区域必须记录：

- 起始偏移：相对 `workspace` 的字节偏移。
- 长度：元素数与字节数。
- 元素类型：`float`、`int32_t`、`uint32_t` 或 byte。
- 对齐：按字节还是按元素向上取整。
- 写入方和读取方。
- 区域末端：`start_bytes + length_bytes`。

最终比较 Host 自定义 workspace 预算与 Kernel 最大区域末端。系统 workspace 应单独列出，不默认可用于弥补自定义区域不足。

## 工作流

### Step 1：定位 Tiling 表面

列出：

1. TilingData 定义文件和注册方式。
2. Host Tiling 入口、平台信息、Shape/属性读取。
3. workspace 申请、`SetBlockDim` 和 tiling key。
4. Kernel 入口、TilingData 解码和 `Init`。
5. 所有读取 TilingData 的 Kernel 位置。

输出：文件清单和调用链，不扩展到无关算子逻辑。

### Step 2：建立字段生命周期矩阵

使用 [Tiling 审查清单](references/tiling-review-checklist.md)，为每个字段记录：

- 定义类型与注释单位。
- Host 赋值表达式及成功路径。
- 是否存在窄化转换。
- Kernel 读取位置。
- 实际消费者：循环、分支、指针偏移、缓冲区长度或调度。
- 未赋值、未读取或读取后不影响行为的证据。

不能仅用文本搜索结果判定字段无用；需要检查宏、模板和生成注册是否可能隐式消费。无法确认时使用 `not_covered`。

输出：字段生命周期矩阵。

### Step 3：审查调度闭环

逐项检查：

- `SetBlockDim(x)` 与 Kernel 使用的 block/core 数是否来自同一计算。
- 每个 block 的起点、结束位置和余数分配是否覆盖完整任务且不重叠。
- 核间同步缓冲区大小是否依据实际 block 数或平台核数。
- Host 设置的 tiling key 是否对应 Kernel 编译实例或显式分支。
- Kernel 模板参数即使未在函数体直接引用，是否可能由编译分发使用；证据不足时不要判死字段。
- 平台信息的类型是否安全传入 TilingData 的窄类型字段。

输出：调度公式、覆盖范围和需要确认的分发行为。

### Step 4：重建 workspace 内存地图

1. 把 Host workspace 公式全部换算为字节。
2. 从 Kernel 第一个 `SetGlobalBuffer` 开始，按真实指针类型换算每个偏移。
3. 记录重叠区域是有意复用、互斥使用还是缺少证据。
4. 计算 Kernel 最大末端。
5. 分开比较：
   - 自定义区域预算。
   - 系统 workspace 大小。
   - 总申请大小。
6. 对不同 block 数、属性开关或 tiling key 重算可能变化的布局。

指针加法单位取决于转换后的类型：

```text
(__gm__ int32_t*)workspace + offset
```

表示 `offset * sizeof(int32_t)` 字节，不能把 `offset` 直接当字节。

输出：逐段内存地图与容量差值。

### Step 5：检查常量、单位与重复真源

重点检查：

- Host 的 `tileLength` 与 Kernel `TILE_LENGTH`。
- byte/block/element 的对齐常量是否名称清楚。
- TilingData 中记录的值是否被 Kernel 读取，还是 Kernel 使用另一个 constexpr。
- `uint32_t` 平台量写入 `uint8_t/uint16_t` 字段的范围依据。
- workspace 公式中的乘法是否在转换到 `size_t` 前完成。
- 注释描述的区域顺序是否与实际偏移一致。

重复常量不等于当前错误。若值目前一致但未来可能分别修改，标记 `duplicated_source_of_truth`。

输出：单位表和重复真源清单。

### Step 6：分级结论

仅使用：

- `closed_loop`：定义、赋值、传递、消费和行为影响证据完整。
- `dead_or_reserved_field`：字段未赋值或未被可见 Kernel 逻辑消费；若可能为预留字段需注明。
- `duplicated_source_of_truth`：同一配置在 Host/TilingData/Kernel 中有多个独立常量。
- `layout_capacity_needs_confirmation`：Host 预算与 Kernel 字节地图存在差值或系统区域边界不明确，尚无运行证据。
- `confirmed_static_mismatch`：类型、单位或偏移有互斥且无需运行即可确认的定义。
- `runtime_confirmed`：实际运行复现并保留了环境、命令和日志。
- `not_covered`：生成代码、编译分发或平台规则缺失，无法静态判断。

不得把 `dead_or_reserved_field` 直接写成缺陷，也不得把容量疑问写成已确认越界。

### Step 7：提出最小改进

按问题类型建议：

- 未消费字段：删除、补赋值/消费，或加注释说明预留目的。
- 重复常量：选择一个真源，增加静态断言，或明确同步检查。
- 单位不清：变量名、注释和公式同时标注 bytes/elements/blocks。
- workspace 风险：先补字节地图或断言，再决定是否调整申请公式。
- tiling key 不透明：补 key 到 Kernel 实例/分支的映射说明。

涉及实际算子代码修改时暂停；本 skill 默认只产出审查报告。

### Step 8：整理 PR-ready 报告

使用 [Tiling 审查报告模板](references/tiling-review-report-template.md) 输出：

- 字段生命周期矩阵。
- 调度和 tiling key 闭环。
- workspace 字节地图。
- 重复真源与单位风险。
- 静态结论、运行状态和未验证项。

## 仓库内算子观察点

### `unique_v3`

- Host 将 `tileLength=8192` 写入 TilingData 并传给 `Init`，Kernel 同时存在独立的 `TILE_LENGTH=8192`；需确认参数是否真正消费。
- workspace 包含排序双缓冲、同步区、block 统计区、counts 和 inverse 临时区；必须统一换算成字节后比较。
- Host 的 block 对齐表达式和 Kernel 的 `(blockNum + 7) / 8 * 8` 使用不同写法，不能只凭注释判断等价。
- `aivNum`、`blockNum` 和 `shortBlockNum` 使用窄类型，需要记录平台范围依据。

### `optimized_transducer`

- `totalPositions`、`vocabSize`、`batchSize`、`usedCoreNum` 直接影响 GM 长度和任务分配，可形成字段闭环。
- TilingData 中的 `tileNum`、`blockSize` 需要检查是否有可见 Kernel 消费。
- Host 申请两个 `totalPositions` 大小的 float workspace，Kernel 分成 alpha/beta 两段，可做字节级闭环验证。
- Host 设置 tiling key，Kernel 使用 `schMode` 模板；编译分发证据不足时标记 `not_covered`，不要仅因函数体未引用模板参数就判定无效。

这些观察点只用于 dry-run，不预设实现有缺陷。

## 错误处理

- 找不到 TilingData 源定义：检查生成文件和注册宏；仍缺失则标记 `not_covered`。
- Host 与 Kernel 使用不同类型指针：全部换算为字节，再比较区域。
- 区域有意复用但生命周期不明：标记 `layout_capacity_needs_confirmation`，请求执行顺序证据。
- tiling key 通过构建系统隐式分发：记录构建入口和缺失证据，不猜测生成行为。
- 无 NPU：运行静态矩阵和公式 dry-run，所有运行结论标记 `NOT_RUN`。
- 发现疑似代码问题：先保留最小公式、文件行号和差值；用户要求修复后再改实现。

## 验证

至少执行：

```bash
git diff --check
rg -n "TILING_DATA|tiling->|tilingData->|SetBlockDim|SetTilingKey|workspace|SetGlobalBuffer" <operator-dir>
rg -n "(A[K]IA|BEGIN [A-Z ]*PRIVATE K[E]Y|GITCODE[_]TOKEN|Authorizatio[n]:|passwor[d]\\s*=|t[o]ken\\s*=|s[e]cret\\s*=)" <changed-paths>
```

对至少一个现有算子完成字段生命周期和 workspace 字节地图 dry-run。公式涉及多个 block 数时，选择能改变对齐结果的代表值。无 NPU 时不能声称运行安全或已复现越界。

## 输出格式

最终返回：

1. Tiling 文件和调用链。
2. 字段生命周期矩阵。
3. blockDim、任务划分和 tiling key 审查。
4. workspace 字节地图与容量差值。
5. 重复真源、单位和窄化风险。
6. 静态分类、运行状态、未验证项和 PR 描述片段。

## 参考资料

- [Tiling 审查清单](references/tiling-review-checklist.md)
- [Tiling 审查报告模板](references/tiling-review-report-template.md)
