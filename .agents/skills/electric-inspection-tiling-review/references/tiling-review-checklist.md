# Tiling 审查清单

读取时机：审查 Host Tiling、TilingData、Kernel 调度或 workspace 布局时读取。

本文件提供字段和内存地图的记录结构，补充 `SKILL.md` 的执行步骤，不替代目标算子源码证据。

## 导航

- [字段生命周期](#字段生命周期)
- [调度闭环](#调度闭环)
- [workspace 字节地图](#workspace-字节地图)
- [常量与单位](#常量与单位)
- [`unique_v3` dry-run 提示](#unique_v3-dry-run-提示)
- [`optimized_transducer` dry-run 提示](#optimized_transducer-dry-run-提示)

## 字段生命周期

| 字段 | 定义类型/单位 | Host 计算 | Host 赋值 | Kernel 读取 | 行为影响 | 结论 |
|---|---|---|---|---|---|---|
| `<field>` | `<uint32_t / elements>` | `<formula>` | `<path:line>` | `<path:line>` | `<loop/offset/branch>` | `<status>` |

逐字段检查：

- 所有 Host 成功路径是否赋值。
- 默认值来自 `memset`、构造器还是显式赋值。
- 是否从宽类型窄化写入。
- Kernel 是读取字段本身，还是使用同名/同值 constexpr。
- 读取后是否真正影响行为。
- 字段是否只为未来 profile、tiling key 或生成工具预留。

## 调度闭环

| 项目 | Host | Kernel | 检查 |
|---|---|---|---|
| block 数 | `SetBlockDim(...)` | `GetBlockNum()` / TilingData | 是否同源 |
| block 索引 | 任务分配公式 | `GetBlockIdx()` | 是否全覆盖且不重叠 |
| 余数 | long/short block 或 sample remainder | 起止偏移 | 是否只分配一次 |
| 同步 | sync workspace 公式 | event/IBSet/IBWait | 容量是否按实际参与核数 |
| tiling key | `SetTilingKey(...)` | 模板/分支/实例 | 映射证据是否存在 |

核对恒等式：

```text
sum(block_work) == total_work
min(block_start) == 0
max(block_end) == total_work
adjacent ranges do not overlap
SetBlockDim == number of scheduled ranges
```

## workspace 字节地图

### Host 预算

| 区域 | Host 表达式 | 元素数 | 元素大小 | 字节数 |
|---|---|---:|---:|---:|
|  |  |  |  |  |

### Kernel 布局

| 区域 | 指针类型 | start 表达式 | start bytes | length 表达式 | length bytes | end bytes |
|---|---|---|---:|---|---:|---:|
|  |  |  |  |  |  |  |

计算规则：

```text
typed_pointer + offset_elements
start_bytes = offset_elements * sizeof(pointer_element_type)
end_bytes = start_bytes + length_elements * sizeof(element_type)
```

检查：

- Kernel 最大 `end_bytes` 是否不大于自定义 workspace 预算。
- 系统 workspace 是否被明确放在自定义区域之后或由框架单独管理。
- Host 与 Kernel 的对齐量是 bytes 还是 elements。
- 属性关闭时区域是否仍被初始化或访问。
- 不同类型的别名访问是否保持相同字节范围。

## 常量与单位

| 概念 | Host 真源 | TilingData | Kernel 真源 | 风险 |
|---|---|---|---|---|
| tile length |  |  |  |  |
| alignment |  |  |  |  |
| core count |  |  |  |  |
| system workspace |  |  |  |  |

推荐单位后缀：

- `_bytes`
- `_elements`
- `_tiles`
- `_blocks`
- `_cores`

## `unique_v3` dry-run 提示

设：

```text
L = ceil_div(totalLength, 8192) * 8192
B8 = ceil_div(blockNum, 8) * 8
B32 = ceil_div(blockNum, 32) * 32
S = blockNum * 32 * 8 + aivNum * 32 + 32
```

复算时至少覆盖 `blockNum=8、16、25、32、33`，因为 Host 与 Kernel 的 block 对齐表达式会在这些位置产生不同字节结果。

需要确认：

- Host 的自定义区域预算是否覆盖 Kernel 最后一个 message 区域。
- 多出的系统 workspace 是否允许被自定义偏移使用。
- `tileLength` Tiling 字段与 Kernel constexpr 是否有单一真源。

所有容量差值在没有框架布局和运行证据时标记 `layout_capacity_needs_confirmation`。

## `optimized_transducer` dry-run 提示

字段生命周期优先检查：

```text
tileNum
blockSize
totalPositions
vocabSize
batchSize
usedCoreNum
```

workspace 闭环：

```text
Host custom bytes = 2 * totalPositions * sizeof(float)
Kernel alpha bytes = totalPositions * sizeof(float)
Kernel beta start = alpha end
Kernel beta end = 2 * totalPositions * sizeof(float)
```

tiling key 检查需要同时查看 Host `SetTilingKey`、Kernel 模板参数和构建生成规则。只看到模板参数未在函数体中使用，不足以判定 key 无效。
