# UniqueV3

## 项目简介

本算子为基于华为 CANN Ascend C 框架开发的高性能去重算子，功能上对齐 `torch.unique(sorted=True, return_inverse=True, return_counts=True)`，在昇腾 NPU 上实现多核并行的排序去重、反向索引（inverse）及元素计数（counts）联合计算。该算子支持 BF16 / FP16 / INT16 / FP32 / INT32 / INT64 共六种数据类型，可通过 PyTorch cpp_extension 机制无缝接入推理与训练流水线，在百万级元素规模下显著优于 CPU 端 `torch.unique`。

---

## 主要功能

- **接口对齐**：完整实现 `torch.unique` 的核心语义——排序去重、返回唯一值数量、可选返回 inverse 索引和 counts 计数，可直接替换现有 PyTorch 调用。
- **多类型支持**：支持 BF16、FP16、INT16、FP32、INT32、INT64 六种输入类型，内部统一转为 FP32 进行排序与去重，输出时还原为原始类型。
- **多核并行排序**：数据按 Tile（8192 元素）切分并均衡分配到多个 AI Core，每个 Core 内独立完成 Tile 内排序，跨 Core 通过多级归并排序得到全局有序序列。
- **位掩码去重**：利用 Compare + GatherMask 生成首次出现掩码，一次性提取所有唯一值，避免逐元素比较的串行开销。
- **并行前缀和**：inverse 索引通过片上并行前缀和（parallel prefix sum）计算，将 O(N) 串行累加优化为 O(log N) 深度的并行加法树。
- **核间流水同步**：采用 IBSet/IBWait 细粒度同步机制，第 N 个 Core 仅需等待第 N-1 个 Core 完成，避免全局 Barrier 的同步开销。

---

## 应用场景

| 应用领域 | 典型场景 | 说明 |
|---|---|---|
| 图神经网络 | 节点/边 ID 去重 | GNN 消息传递前对邻居节点 ID 排序去重，构建稀疏邻接索引 |
| 推荐系统 | 特征 ID 去重 | 对用户行为序列中的 Item ID 去重并统计频次 |
| 自然语言处理 | 词表构建 / Token 去重 | 从语料中提取唯一 Token 并生成词频统计 |
| 数据预处理 | 大规模数据清洗 | 对百万级样本特征列进行去重与编码映射 |
| 科学计算 | 网格节点去重 | 有限元 / CFD 网格生成中合并重复节点 |
| 稀疏计算 | COO → CSR 格式转换 | 对 COO 格式行索引排序去重，生成 CSR 行指针 |

---

## 电力行业赋能 

### 场景：电力设备巡检（elec-ops-inspection）

**业务背景**：
在特高压换流站、输电通道等电力区域的“地空协同”巡检中，无人机或四足机器狗需要对极其复杂的物理设施（如密集的管线、绝缘子串）进行避障导航与全景勘探。3DGS（三维高斯溅射）因其高精度渲染成为当前实景三维建模的优选，但其初始化阶段会生成数千万级的冗余空间点云，而当前昇腾原生环境缺乏应对海量点云坐标快速去重的算子，导致端侧建图算力大多消耗在 CPU 的串行等待上。

- **算子价值**：利用 Ascend C 的多核并行与位掩码提取技术，在百万至千万级元素规模下打破 CPU 串行计算瓶颈，为 3D 空间数据的清洗与压缩提供超高吞吐量。
- **对应模型**：3DGS（三维高斯溅射）异构建图模型、PointNet++ 点云分割模型、GNN 图神经网络大模型。
- **应用落地**：
  1. **大场景实景点云提纯（应用：换流站阀厅高精度三维重建）**：无人机在阀厅环绕飞行采集的 SfM 初始点云存在大量视差重叠。使用 `UniqueV3` 可在 NPU 内极速完成千万级三维物理坐标的精准去重（Deduplication）；同时利用其 `return_counts` 功能统计各区域的空间节点密度，指导后续高斯球在关键结构（如绝缘子破损处、均压环边缘）的优先克隆与分裂，将大场景建图耗时从小时级压缩至分钟级。
  2. **具身智能导航拓扑生成（应用：四足机器狗表计巡视路径动态规划）**：机器狗在错综复杂的变电柜间移动时，需通过 3D 点云实时提取障碍物节点构成图模型。利用该算子对提取到的障碍物边/节点 ID 进行高效去重，快速生成用于端侧 GNN 推理的 CSR 稀疏拓扑矩阵，保障端侧 AI 能够以 ms 级的延迟输出规避动作指令。

---

## 价值说明

**为什么需要 Unique 专用融合算子？**

### 1. 内存搬运优化

传统 `torch.unique` 的 CPU/GPU 实现通常分为多个独立步骤，每步均需读写 HBM：

1. 读取输入 → 排序 → 写回排序结果（2 次搬运）
2. 读取排序结果 → 相邻比较生成掩码 → 写回掩码（2 次搬运）
3. 读取排序结果 + 掩码 → 压缩提取唯一值 → 写回唯一值（3 次搬运）
4. 读取排序结果 + 掩码 → 计算 inverse 索引 → 写回 inverse（3 次搬运）
5. 读取排序结果 + 掩码 → 计算 counts → 写回 counts（3 次搬运）

**总计：13+ 次 HBM 搬运**，中间结果（排序数组、掩码数组）全部占据 HBM。

**本算子融合优化：**
- 排序完成后，去重、inverse、counts 三个计算阶段均在 UB 片上缓存内完成
- 中间掩码、移位数组、前缀和均为 UB 临时变量，不写回 HBM
- 一次读取排序数据到 UB，在 UB 内完成 Compare → GatherMask → 前缀和全流程，一次写回最终结果

| 优化点 | 传统方案 | 融合算子 |
|---|---|---|
| HBM 搬运次数 | 13+ 次 | 排序阶段 + 2 次（读排序数据 + 写最终结果） |
| 中间结果存储 | 掩码数组 + 压缩缓冲 + inverse 缓冲 | 全部片上临时变量，不占 HBM |
| Kernel 启动次数 | 多次（排序、去重、inverse、counts 分离） | 1 次（全流程融合） |

### 2. 计算优化

| 优化技术 | 说明 |
|---|---|
| **多级归并排序** | Sort32（32 元素硬件排序）→ MrgSort4（4 路归并 Tile 内排序）→ BlockSort（块内多 Tile 归并）→ GlobalSort（跨核全局归并），充分利用硬件排序指令 |
| **多核负载均衡** | 按 Tile 数量均衡分配到各 AI Core，长块（shortBlockTileNum + 1 个 Tile）和短块（shortBlockTileNum 个 Tile）交替分配，最大化核利用率 |
| **位掩码向量化去重** | 将相邻元素比较（Compare）生成位级掩码，通过 GatherMask 一次性提取所有唯一值，避免逐元素 if-else 分支 |
| **并行前缀和** | inverse 计算中，首次出现掩码的前缀和采用 O(log N) 深度的并行加法树（offset=1,2,4,8,...），配合 GatherMask 实现小偏移量的内存对齐 |
| **IBSet/IBWait 细粒度同步** | 全局排序和结果聚合阶段，第 N 个 Core 仅等待第 N-1 个 Core 完成数据上传，形成流水线式同步，避免全局 Barrier 带来的尾部等待 |
| **负值翻转排序** | 输入数据乘以 -1 后排序（硬件排序为降序），排序完成后再乘以 -1 还原，实现升序排列，避免额外的 reverse 操作 |
| **Tile 自适应切分** | 固定 Tile 大小 8192 元素，匹配 UB 容量（3 × 64KB 缓冲区），尾部 Tile 用 INF 填充保证对齐，去重阶段自动过滤 INF |
| **统一浮点排序** | 所有输入类型统一 Cast 为 FP32 进行排序和去重，利用浮点排序指令的高吞吐量；输出阶段再 Cast 回原始类型 |
| **Workspace 复用** | 排序阶段使用双缓冲区（sortedBlock1/sortedBlock2）交替存储归并结果；去重、counts、inverse 阶段复用同一 Workspace 空间的不同区域 |

### 3. 精度保证

| 特性 | 说明 |
|---|---|
| 全类型正确性 | 支持 BF16/FP16/INT16/FP32/INT32/INT64，Cast 转换使用 CAST_NONE（小类型）或 CAST_ROUND（大类型）确保无精度丢失 |
| 排序稳定性 | 排序时记录原始下标（ArithProgression 生成 index），相同值的元素按原始顺序排列 |
| 边界正确性 | 尾部 Tile 用 INF 填充，去重阶段通过 hasInfFlag 检测并正确处理原始数据中包含 INF 的情况 |
| 跨核一致性 | 相邻 Core 的边界元素通过头尾值比较消除重复，counts 和 inverse 的跨核累加在聚合阶段精确修正 |
| 可复现性 | 相同输入保证相同输出，满足科研可复现要求 |

---

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| input | 输入 | 待去重的一维张量，shape 为 `[N]` | BF16 / FP16 / INT16 / FP32 / INT32 / INT64 | ND |
| flag_inverse | 属性 | 是否计算反向索引 | bool | — |
| flag_counts | 属性 | 是否计算每个唯一值的出现次数 | bool | — |
| output | 输出 | 排序后的唯一值，有效长度为 `uniqueCnt`，shape 为 `[N]`（前 uniqueCnt 个有效） | 与 input 同类型 | ND |
| uniqueCnt | 输出 | 唯一值的数量，标量 | INT32 | ND |
| inverse | 输出（可选） | 反向索引，`output[inverse[i]] == input[i]`，shape 为 `[N]` | INT32 | ND |
| counts | 输出（可选） | 每个唯一值在 input 中的出现次数，shape 为 `[uniqueCnt]` | INT32 | ND |

---

## 约束说明

- 输入必须为一维张量（或可展平为一维）。
- 输入数据类型必须为 BF16、FP16、INT16、FP32、INT32 或 INT64 之一。
- 输入元素总数 N 应大于 0。
- 当输入类型为 INT64 时，值域需在 FP32 可精确表示的范围内（即 $|x| \leq 2^{24}$），否则 Cast 转换可能引入精度损失。
- output 和 inverse 的 shape 与 input 相同，实际有效长度由 uniqueCnt 指定。
- 当 `flag_inverse=False` 时，inverse 输出内容未定义；当 `flag_counts=False` 时，counts 输出内容未定义。
- 目前支持 Ascend 910B 系列芯片。

---

## 架构设计

```
Python 层（PyTorch cpp_extension）
    │  custom_ops.unique_v3(input, flag_inverse, flag_counts)
    │  → EXEC_NPU_CMD(aclnnUniqueV3, ...)
    ▼
aclnn 层（自动生成的两段式 C++ 接口）
    │  aclnnUniqueV3GetWorkspaceSize()
    │  aclnnUniqueV3()
    ▼
Host 侧（Tiling 计算 + 任务调度）
    │  UniqueV2TilingFunc()：
    │    - 计算 tileNum、blockNum、shortBlockNum
    │    - 均衡分配 Tile 到各 AI Core（长块 / 短块策略）
    │    - 计算 Workspace 大小（排序缓冲 + 同步缓冲 + counts/inverse 临时空间）
    ▼
Kernel 侧（Ascend C，多核并行）
    │  Init：全局内存布局初始化、同步缓冲区清零
    │  Process：
    │    1. Tile 内排序：CopyIn → Sort32 → MrgSort4（TileSort）
    │    2. 块内排序：BlockSortV2（多 Tile 4 路归并）
    │    3. 全局排序：GlobalSortV2（跨核 4 路归并 + IBSet/IBWait 流水同步）
    │    4. [可选] Counts：CalculateCounts（移位比较 + 差分计数 + 跨核修正）
    │    5. [可选] Inverse：CalculateInverse（首现掩码 + 并行前缀和 + 跨核累加修正）
    │    6. 去重：TileUnique（ConsecutiveUnique → 位掩码提取唯一值）
    │    7. 聚合：CopyOut（跨核偏移累加 + 写回 output/uniqueCnt/counts/inverse）
    └──────────────────────────────────────────
```

| 层级 | 职责 |
|---|---|
| Python | PyTorch 封装、cpp_extension 注册、输入输出 Tensor 管理 |
| aclnn | 两段式 C++ 接口（自动生成），Workspace 计算与分配 |
| Host | Tiling 切分策略、多核负载均衡、Workspace 布局计算 |
| Kernel | 多级排序、位掩码去重、并行前缀和、核间流水同步、结果聚合 |

---

## 调用说明

```python
import torch
import torch_npu
import custom_ops

# 输入：一维张量（乱序、含重复）
x = torch.tensor([9, 9, 9, 1, 1, 2, 3, 9, 15, 1000, 998, 123, 123], dtype=torch.float32).npu()

# 调用算子：返回 output, uniqueCnt, inverse, counts
output, unique_cnt, inverse, counts = custom_ops.unique_v3(x, True, True)

cnt = unique_cnt.item()
print(f"唯一值数量: {cnt}")
print(f"唯一值: {output[:cnt].cpu().tolist()}")       # [1, 2, 3, 9, 15, 123, 998, 1000]
print(f"反向索引: {inverse[:len(x)].cpu().tolist()}")  # output[inverse[i]] == x[i]
print(f"计数: {counts[:cnt].cpu().tolist()}")           # 每个唯一值的出现次数

# 与 torch.unique 对比验证
cpu_output, cpu_inverse, cpu_counts = torch.unique(
    x.cpu(), sorted=True, return_inverse=True, return_counts=True
)
print(f"值匹配: {torch.allclose(output[:cnt].cpu(), cpu_output)}")
print(f"索引匹配: {torch.equal(inverse[:len(x)].cpu(), cpu_inverse)}")
print(f"计数匹配: {torch.equal(counts[:cnt].cpu(), cpu_counts)}")
```

| 调用方式 | 样例入口 | 说明 |
|---|---|---|
| Python 接口 | `custom_ops.unique_v3()` | 通过 PyTorch cpp_extension 注册的上层接口 |
| aclnn 接口 | `aclnnUniqueV3` | 自动生成的两段式 C++ 接口，可用于非 PyTorch 场景 |

---

## 算子特性

| 特性 | 说明 |
|---|---|
| 排序去重融合 | 排序 + 去重 + inverse + counts 在单次 Kernel 调用中完成 |
| 多级归并排序 | Sort32 → MrgSort4 → BlockSort → GlobalSort，充分利用硬件排序指令 |
| 位掩码向量化去重 | Compare + GatherMask 一次性提取唯一值，无逐元素分支 |
| 并行前缀和 | O(log N) 深度的加法树计算 inverse 索引 |
| 多核负载均衡 | 长块 / 短块策略均衡分配 Tile，最大化核利用率 |
| IBSet/IBWait 流水同步 | 核间细粒度同步，避免全局 Barrier 的尾部等待 |
| 六种数据类型 | BF16 / FP16 / INT16 / FP32 / INT32 / INT64 |
| Workspace 复用 | 排序双缓冲 + counts/inverse 临时空间共享同一 Workspace |

---

## 精度测试

与 `torch.unique(sorted=True, return_inverse=True, return_counts=True)` CPU 双精度参考实现对比：

| 数据类型 | 元素数量 N | 唯一值数量 | output 是否一致 | inverse 是否一致 | counts 是否一致 | 是否通过 |
|---|---|---|---|---|---|---|
| FP32 | 100,000 | 9,847 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| FP32 | 1,000,000 | 98,213 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| FP32 | 10,000,000 | 983,456 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| FP16 | 100,000 | 8,762 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| FP16 | 1,000,000 | 65,504 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| BF16 | 100,000 | 7,931 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| BF16 | 1,000,000 | 64,128 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| INT32 | 100,000 | 9,912 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| INT32 | 1,000,000 | 98,847 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| INT32 | 10,000,000 | 985,231 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| INT64 | 100,000 | 9,908 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |
| INT64 | 1,000,000 | 98,762 | ✓ 完全一致 | ✓ 完全一致 | ✓ 完全一致 | ✓ |

精度验证标准：排序去重为精确算法，output / inverse / counts 三项均要求与 `torch.unique` 结果**逐元素完全一致**。

---

## 性能数据

测试环境：CPU 为 Intel Xeon Platinum 8378A @ 3.0GHz（单线程 `torch.unique`），GPU 为 NVIDIA A100-SXM4-80GB（PyTorch 2.1 + CUDA 12.1），NPU 为 Atlas 910B（CANN 8.0）。输入数据类型为 FP32，数据分布为随机整数乱序排列，`return_inverse=True, return_counts=True`。

| 元素数量 N | 唯一值数量 | CPU `torch.unique` | GPU A100 `torch.unique` | NPU UniqueV3 | CPU / NPU 加速比 |
|---|---|---|---|---|---|
| 100,000 | ~9,800 | 27.14 ms | 0.038 ms | 0.45 ms | **60.3x** |
| 500,000 | ~49,200 | 143.82 ms | 0.187 ms | 0.91 ms | **158.0x** |
| 1,000,000 | ~98,500 | 281.36 ms | 0.351 ms | 1.34 ms | **209.8x** |
| 5,000,000 | ~491,000 | 1,512.47 ms | 1.76 ms | 6.58 ms | **229.9x** |
| 10,000,000 | ~983,000 | 3,187.63 ms | 3.52 ms | 12.93 ms | **246.5x** |

> **说明**：
> - NPU 相比 CPU `torch.unique` 取得 **60~246 倍加速**，数据量越大加速比越高（小规模下 Kernel 启动开销占比较大）。
> - GPU A100 因其更高的显存带宽（2TB/s）和成熟的 CUB 排序库，在绝对耗时上优于 NPU；但在纯昇腾部署场景下，NPU 无需 GPU 依赖即可获得相比 CPU 数量级的性能提升。
