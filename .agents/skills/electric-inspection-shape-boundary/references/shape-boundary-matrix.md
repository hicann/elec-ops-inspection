# Shape 边界矩阵

读取时机：分析算子 Shape、Tiling、对齐、调度或多输入关系时读取。

本文件帮助把源码表达式转换成可复核的边界用例。核心结论和分级规则以 `SKILL.md` 为准。

## 导航

- [公式记录表](#公式记录表)
- [边界类型](#边界类型)
- [关系型 Shape 表](#关系型-shape-表)
- [`unique_v3` 静态代入示例](#unique_v3-静态代入示例)
- [`optimized_transducer` 静态关系示例](#optimized_transducer-静态关系示例)

## 公式记录表

每个派生量填写一行：

| 派生量 | 公式 | 输入变量 | C++ 类型 | 单位 | 前置条件 | 消费位置 |
|---|---|---|---|---|---|---|
| `<tileNum>` | `<ceil_div(N, tileLength)>` | `<N, tileLength>` | `<uint32_t>` | `<tile>` | `<N > 0, tileLength > 0>` | `<Kernel loop>` |

检查问题：

- 除数、取模数是否可能为零？
- 乘法和加法在哪个类型中执行？
- 先乘后转型还是先转型后乘？
- 元素数和字节数是否混用？
- Host 与 Kernel 使用相同的对齐公式吗？
- 公式结果是否被窄化到 `uint8_t`、`uint16_t` 或 `int32_t`？

## 边界类型

### Rank 与零维

| 边界 | 建议用例 |
|---|---|
| rank 不足 | 比最小 rank 少 1 |
| rank 恰好合法 | 最小合法 rank |
| 多余 rank | 比文档声明多 1，确认拒绝还是展平 |
| 零维 | 每个维度分别置 0 |
| 空 Shape | 框架允许时测试空张量 |

### Tile 与尾块

对固定 tile 大小 `T`，优先选择：

```text
1
T - 1
T
T + 1
2T - 1
2T
2T + 1
```

同时记录：

- `tileNum`
- `tailLength`
- `blockNum`
- 每个 block 的 tile 数
- 最后一个 block 是否进入 tail 分支

### Core 切换

若 `blockNum = min(tileNum, coreNum)`：

```text
tileNum = coreNum - 1
tileNum = coreNum
tileNum = coreNum + 1
```

平台 `coreNum` 未知时保留符号表达式，不猜具体数值。

若 `usedCoreNum = min(B, coreNum)`，还需检查 `B=0` 是否有单独保护。

### 对齐

对对齐单位 `A`：

```text
A - 1
A
A + 1
2A - 1
2A
2A + 1
```

记录对齐后的长度、额外 padding、DataCopy 长度和 GM/UB 偏移。

### 整数范围

对 Shape 乘积检查：

- 每个维度的读取类型。
- 中间乘积类型。
- 目标字段类型。
- `max / other_factor` 附近的最小溢出用例。
- workspace 字节数是否可能在转换前溢出。

不要实际申请超大张量来证明算术风险；可先用符号或安全脚本计算阈值。

## 关系型 Shape 表

| 关系 ID | 表达式 | 合法最小样例 | 单项非法样例 | 保护位置 | 结论 |
|---|---|---|---|---|---|
| `REL-001` | `<len(lengths) == B>` |  |  |  |  |

常见关系：

- 所有 batch 维相等。
- `0 < length[i] <= padded_dimension`。
- packed 长度等于逐样本有效面积之和。
- 输出容量不小于 Kernel 最大写入偏移。
- 动态有效长度不超过分配 Shape。

## `unique_v3` 静态代入示例

当前可见 Host 公式：

```text
tileLength = 8192
tileNum = ceil_div(totalLength, tileLength)
blockNum = min(tileNum, aivNum)
shortBlockTileNum = tileNum / blockNum
longBlockNum = tileNum % blockNum
shortBlockNum = blockNum - longBlockNum
```

建议 dry-run 表：

| N | tileNum | tailLength | 观察点 |
|---:|---:|---:|---|
| 0 | 0 | 0 | `blockNum` 可能为 0，检查整除前保护 |
| 1 | 1 | 1 | 最小合法输入 |
| 8191 | 1 | 8191 | 单 tile 尾块 |
| 8192 | 1 | 0 | 无尾块 |
| 8193 | 2 | 1 | 第二 tile 与尾块 |

这只是公式代入。没有运行证据时，`N=0` 路径应标记 `unguarded_static_risk`，不能写成已确认崩溃。

## `optimized_transducer` 静态关系示例

符号约定：

- `logits = [P, V]`
- `targets = [B, Umax]`
- `logit_lengths = [B]`
- `target_lengths = [B]`

待验证关系：

```text
V > 0
B > 0
len(logit_lengths) == B
len(target_lengths) == B
targets.dim(0) == B
P == Σ(logit_lengths[i] * (target_lengths[i] + 1))
0 < logit_lengths[i]
0 <= target_lengths[i] <= Umax
```

建议最小非法样例每次只破坏一个关系，以便定位拒绝发生在 Host、框架还是 Kernel。
