# reduce-all-ascend

**项目简介**

本项目为基于华为 CANN 计算框架开发的布尔归约算子，实现与 `torch.all` 功能完全对齐的 ReduceAll 算子。算子支持对任意维度的布尔张量（`bool` / `int8`）沿指定轴执行逻辑与归约（`∧`），支持多轴归约与全归约两种模式，并通过轴融合、多模式分发、AtomicMin 多核写合并等技术在昇腾 910B NPU 上实现高效计算。

---

**主要功能**

- **接口对齐**：完整实现 `torch.all(input, dim, keepdim)` 的全部参数语义，支持单轴、多轴、全归约三种调用形式，可直接替换 PyTorch 原生调用。
- **轴融合优化**：将输入维度中连续的 Reduce 轴（R）与 Keep 轴（K）合并，消除冗余的下标计算，并根据融合后的维度模式自动选择专用 Kernel 分支。
- **六种归约模式**：针对融合后常见的轴排列模式（FullReduce / KR / RK / RKR / KRK / General）提供独立优化实现，各分支针对对应的内存访问模式做专项加速。
- **AtomicMin 多核安全写回**：各核将归约结果写入独立的 WorkGM 缓冲区，最终通过 `SetAtomicMin` 原子操作安全合并到输出 GM，避免多核写冲突。
- **Double Buffer 流水线**：利用 AscendC 双缓冲机制，在计算当前 Tile 的同时后台预取下一 Tile 数据，隐藏 HBM 访问延迟，提升硬件利用率。
- **数据类型支持**：支持 `DT_BOOL` 与 `DT_INT8` 两种输入类型，内部统一转换为 `half` 参与向量计算。

---

**应用场景**

| 应用领域 | 典型场景 | 说明 |
|---|---|---|
| 模型推理 | 掩码有效性校验 | 批量检查 attention mask 中各序列的 padding 掩码是否全部有效 |
| 数据预处理 | 批量条件过滤 | 对大规模布尔条件矩阵按行/列归约，快速筛选满足所有条件的样本 |
| 训练监控 | 梯度健康检查 | 检测梯度张量中是否存在 NaN/Inf（转布尔后全归约） |
| 图神经网络 | 边存在性判断 | 对邻接矩阵按节点维度归约，判断节点间连通性 |
| 强化学习 | 终止状态判断 | 对 batch done 标记向量做全归约，判断当前 episode 是否全部结束 |

---

**昇腾原生 × 电力行业赋能**

**场景：电力负荷预测（elec-ops-prediction）**

**业务背景**：
区域电网实际业务中，针对特定台区或园区的超短期（分钟级）负荷预测直接关系到换流站的功率动态调节与储能调度。这类预测需融合传感器实时采集的温湿度、光照、电流等高维特征矩阵。然而，边缘采集设备极易受电磁干扰，导致时序数据常出现断点或跳变，前置的“数据有效性校验”往往会卡住整个 NPU 算力流水线，导致推理延迟超标。

- **算子价值**：提供与 `torch.all` 语义对齐的硬件级加速，将传统 HBM 访存次数从 6 次降低至 2 次，消除时序矩阵在 NPU 上的逻辑归约瓶颈，为模型推理提供更高效的数据吞吐支持。
- **对应模型**：LSTM / GRU 时序网络、Informer / Autoformer 长时序预测大模型、GNN-Transformer 电网空间拓扑模型。
- **应用落地**：
  1. **传感器异常的极速熔断（应用：关口表计分钟级功率动态预测）**：在分钟级预测流水线中，将当前时间窗口内多维特征的“是否缺失/异常”状态映射为布尔矩阵，利用 `ReduceAll` 执行多轴并行归约。一旦判定核心特征全为 `False`，立即在 NPU 侧触发短路退出（Short-circuit），阻止脏数据喂入 LSTM 导致功率预测值剧烈震荡，并瞬间调起上层应用的历史均值平滑插值策略。
  2. **大模型高维特征的掩码过滤（应用：跨区域气象-负荷联合中长期预测）**：在使用 Informer 等大模型的注意力机制处理长达数周的时序特征时，需对大量零值或无效时间步进行有效性校验（Padding Mask）。利用本算子的轴融合优化（Fuse Axes），能极速归约剔除无效的冷空气/降雨影响因子序列，大幅提升 Transformer 结构对高维时序特征的注意力计算效率。


---

**优化说明**

**1. 内存搬运优化**

传统分步 `torch.all` 实现在 NPU 上通常被拆解为多个独立 Kernel 依次调用，典型流程如下：

- 读取输入布尔张量 → 类型转换为浮点 → 写回中间张量（2 次搬运）
- 读取中间张量 → 执行 ReduceMin/ReduceSum → 写回归约结果（2 次搬运）
- 读取归约结果 → 与阈值比较 → 写回最终布尔输出（2 次搬运）

总计约 **6 次** HBM 搬运，且存在多个中间张量占据 HBM。

本算子融合后的优化路径：一次从 HBM 读取输入数据到片上 UB，在 UB 内完成 Cast → Abs → ReduceMin 全流程，直接将结果写回 WorkGM，最后一次 AtomicMin 合并到输出。**总计 2 次 HBM 搬运**，中间结果全部驻留片上。

| 优化点 | 传统方案 | 融合算子 |
|---|---|---|
| HBM 搬运次数 | 6 次 | 2 次 |
| 中间张量数量 | 2 个（浮点中间、归约结果） | 0 个（全部片上完成） |
| 多核写冲突处理 | 需要额外同步 Barrier | AtomicMin 原子写，无需额外同步 |

**2. 计算优化**

| 优化技术 | 说明 |
|---|---|
| 轴融合 | 将连续同类型轴（R-R 或 K-K）合并为单轴，从 N 维退化为最多 3 维，大幅降低下标计算开销 |
| 模式专用分支 | 根据融合后的轴排列（KR/RK/RKR/KRK）选择专用计算路径，避免 General 路径的逐元素坐标映射开销 |
| 快速跳过优化 | 每个 Tile 先做 ReduceMin 预检，若整块 Tile 全为非零（全 True）则直接跳过，无需逐元素扫描 |
| 短路退出 | FullReduce 模式下一旦检测到第一个 False 元素立即终止遍历，无需扫描剩余数据 |
| Tile 自适应分块 | 根据 UB 容量自动计算 tile 面积，公式为 `tileSize = AlignDown((ubSize - FIXED_EXPENSES) / UB_BUFFER_FACTOR, BLOCK_SIZE)`，充分利用片上缓存 |
| Double Buffer | `BUFFER_NUM=2` 双缓冲流水，DMA 搬运与向量计算并发执行，隐藏内存访问延迟 |

核间负载均衡的分配策略如下，设总输入元素数为 $N$，使用核数为 $C$，则：

$$\text{elementsPerCore} = \lfloor N / C \rfloor, \quad \text{largeCoreCount} = N \bmod C$$

前 `largeCoreCount` 个核各处理 `elementsPerCore + 1` 个元素，其余核各处理 `elementsPerCore` 个元素，实现严格均衡。

**3. 精度保证**

| 特性 | 说明 |
|---|---|
| 初始化保护 | WorkGM 与 OutputGM 均初始化为全 1（True），AtomicMin 只能将其改为 0（False），确保未被覆盖的位置保持正确初始值 |
| AtomicMin 原子性 | 多核并发写同一输出位置时通过硬件原子操作保证结果正确，无竞争条件 |
| SyncAll 同步屏障 | 初始化 OutputGM 后显式调用 `SyncAll()`，确保所有核在开始计算前看到一致的初始值 |
| Cache 刷新 | WorkGM 写回前调用 `DataCacheCleanAndInvalid` 强制刷出缓存行，防止 SetValue 写入遗留在 L2 Cache 中未落地到 GM |
| 零维张量处理 | 显式处理含零维输入（hasZeroDim）的退化情况，输出符合 PyTorch 语义 |

---

**参数说明**

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| self | 输入 | 待归约的布尔张量，任意维度 | DT_BOOL / DT_INT8 | ND |
| dim | 属性（可选） | 指定归约的轴列表；为空时执行全归约 | ListInt（int64） | — |
| keepdim | 属性（可选） | 是否保留归约后的维度，默认 false | Bool | — |
| out | 输出 | 归约结果张量 | DT_BOOL / DT_INT8 | ND |

**约束说明**

- 输入张量数据类型必须为 `DT_BOOL` 或 `DT_INT8`，输出类型与输入保持一致。
- `dim` 属性为可选，不传或传空列表时默认对所有维度执行全归约。
- `dim` 中的轴值支持负数索引，负值会自动归一化为 `axis + rank`，越界时返回错误。
- 融合后的维度数 `fusedDimCount` 不得超过 `MAX_DIMS`（编译期常量，通常为 8）。
- 当前仅支持 Ascend 910B AI Core 配置，其他硬件需补充 `AddConfig`。
- 输入张量元素总数为 0（含零维）时走特殊路径，直接输出空张量或标量 True，不启动向量计算。

---

**架构设计**

```
aclnn 层（C++ 两段式接口）
    │  aclnnReduceAllGetWorkspaceSize()
    │  aclnnReduceAll()
    ▼
Host 侧（Tiling，reduce_all_tiling.cpp）
    │  GetReduceAxesFromAttr()    —— 读取 dim 属性，归一化轴列表
    │  ParseAndValidate()         —— 计算 totalInputSize，检测零维
    │  FuseAxes()                 —— 连续同类轴融合，得到 fusedDims/fusedIsReduce
    │  ComputeFusedInputStrides() —— 计算融合维的 inputStrides
    │  ComputeOutputMetaByFused() —— 计算 totalOutputSize 和 outputStrides
    │  DetermineReduceMode()      —— 根据轴模式选择 ReduceMode（6 种）
    │  ComputeMaxTileSize()       —— 根据 UB 容量计算最大 Tile 字节数
    │  ComputeCoreAllocation()    —— 均匀分核，计算 elementsPerCore / largeCoreCount
    ▼
Kernel 侧（AscendC，reduce_all.h）
    │  Init()
    │    ├── 绑定 inputGM / outputGM / workGM 全局内存
    │    ├── 初始化 outputGM 和 workGM 为全 1（True），SyncAll 同步
    │    └── InitBuffer：inputQueue(×2) / workBuf / tmpBuf / minBuf
    │
    └── Process()
          ├── MODE_FULL_REDUCE → ProcessAllReduce()   全归约，短路退出
          ├── MODE_KR          → ProcessKR()           [Keep][Reduce] 模式
          ├── MODE_RK          → ProcessRK()           [Reduce][Keep] 模式
          ├── MODE_RKR         → ProcessRKR()          [R][K][R] 三段模式
          ├── MODE_KRK         → ProcessKRK()          [K][R][K] 三段模式
          └── MODE_GENERAL     → ProcessGeneral()      通用坐标映射模式
                │
                └── CopyWorkGmToOutput()
                      DataCacheCleanAndInvalid → DataCopyPad(workGM→UB)
                      → DataCopy(UB→UB) → SetAtomicMin + DataCopyPad(UB→outputGM)
```

| 层级 | 职责 |
|---|---|
| aclnn | 两段式 C++ 接口，workspace 计算，算子注册 |
| Host/Tiling | 轴解析、轴融合、模式选择、核间均匀分配、Workspace 计算 |
| Kernel/Init | GM 绑定、WorkGM 初始化、UB Buffer 分配 |
| Kernel/Process | 六模式分发、Tile 分块遍历、ReduceMin 快速跳过、AtomicMin 写回 |

---

**六种归约模式详解**

ReduceAll 算子在 Host 侧完成轴融合后，根据融合结果中 R/K 轴的排列自动选择以下六种专用模式之一。

**MODE_FULL_REDUCE（全归约）** 是输入所有维度均为归约轴的情况，输出单个标量。该模式采用短路策略，遍历过程中一旦发现第一个 False 元素即立刻中止，无需扫描剩余数据，时间复杂度最优情况为 $O(1)$，最差情况为 $O(N)$。

**MODE_KR（[Keep][Reduce]）** 对应融合后的二维布局 `[outerK][innerR]`，即先保留维再归约维。每个 K 位置对应 `innerR` 个连续元素，归约结果写入 `workGm[k]`。该模式对每个 Tile 先做全局 ReduceMin 预检，若整块均非零则跳过，否则按 K 分段独立归约。

**MODE_RK（[Reduce][Keep]）** 对应融合后的二维布局 `[outerR][innerK]`，即先归约维再保留维。每个 K 位置的元素在内存中不连续（步长为 `innerK`），因此按 K 列逐步扫描，对每个 K 列内的 Tile 元素做下标映射后写回 `workGm[k]`。

**MODE_RKR（[R1][K][R2]）** 对应融合后的三维布局 `[outerR1][middleK][innerR2]`，输出维度仅为 `middleK`。该模式通过公式 `k = (globalIdx % (K×R2)) / R2` 直接从全局下标反推 K 坐标，同时维护缓存变量 `lastK` 避免对同一输出位置的重复 GM 读写。

**MODE_KRK（[K1][R][K2]）** 对应融合后的三维布局 `[outerK1][middleR][innerK2]`，输出维度为 `K1 × K2`，输出下标为 `outputIdx = k1 × innerK2 + k2`。通过双缓存变量 `(lastK1, lastK2)` 去除对同一输出位置的重复操作。

**MODE_GENERAL（通用模式）** 处理不符合上述任何模式的任意维度组合，通过 `GlobalIdxToOutputIdx()` 函数按融合维逐级做除余映射，将输入全局下标转换为输出下标，适用性最广但性能相对较低。该模式还针对末尾为 R 轴的情况做了额外优化：检测到 False 后跳过当前 R 段剩余元素，减少无效坐标计算。

---

**调用说明**

```cpp
// C++ aclnn 接口调用示例
#include "aclnn_reduce_all.h"

// 第一段：获取 workspace 大小
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnReduceAllGetWorkspaceSize(
    selfTensor,       // 输入布尔张量
    dimArray,         // 归约轴列表 (aclIntArray*)
    keepdim,          // bool
    outTensor,        // 输出张量
    &workspaceSize,
    &executor
);

// 第二段：分配 workspace 并执行
void* workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
aclnnReduceAll(workspace, workspaceSize, executor, stream);
```

```python
# Python 等价调用（PyTorch 语义对齐）
import torch

x = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

# 全归约
result = torch.all(x)           # tensor(False)

# 按列归约
result = torch.all(x, dim=0)    # tensor([True, False])

# 按行归约，保留维度
result = torch.all(x, dim=1, keepdim=True)  # tensor([[True], [False]])
```

| 调用方式 | 样例入口 | 说明 |
|---|---|---|
| aclnn 接口 | `test_aclnn_reduce_all` | 通过 `aclnnReduceAll` 直接调用底层 Kernel |
| PyTorch 算子注册 | `torch.ops.npu.reduce_all` | 通过 NPU 算子注册后在 PyTorch 中透明调用 |

---

**测试数据**

| 测试用例 | 输入形状 | 归约轴 | 输出形状 | CPU 耗时 (ms) | NPU 耗时 (ms) | 加速比 | 结果 |
|---|---|---|---|---|---|---|---|
| 2D 全归约 | [4, 3000000] | [0, 1] | [1] | 2588.889 | 0.581 | 4452.47x | ✓ |
| 2D 全归约 | [4, 3000000] | [0, 1] | [1] | 2584.791 | 0.162 | 15927.97x | ✓ |
| 2D 全归约 | [4, 3000000] | [0, 1] | [1] | 2585.597 | 0.136 | 18969.90x | ✓ |
| 2D 全归约 | [4, 3000000] | [0, 1] | [1] | 2588.517 | 0.151 | 17157.27x | ✓ |
| 2D 单轴归约 | [4, 3000000] | [1] | [4] | 2433.505 | 0.161 | 15113.06x | ✓ |
| 3D 中间轴归约 | [4, 30000, 5] | [1] | [4, 5] | 121.444 | 2.090 | 58.11x | ✓ |
| 3D 末尾轴归约（小张量） | [4, 3, 5] | [2] | [4, 3] | 0.022 | 0.082 | 0.27x | ✓ |
| 4D 单轴归约 | [2, 4, 500, 200] | [3] | [2, 4, 500] | 162.932 | 0.597 | 273.12x | ✓ |
| 4D 多轴归约 | [2, 4, 500, 200] | [2, 3] | [2, 5] | 173.010 | 0.077 | 2253.32x | ✓ |
| 4D 单轴归约 | [2, 4, 5, 2000] | [1] | [2, 5, 2000] | 21.311 | 1.793 | 11.88x | ✓ |
| 4D 单轴归约 | [2, 400, 5, 2] | [2] | [2, 400, 2] | 2.082 | 0.415 | 5.01x | ✓ |
| 4D 多轴归约 | [2, 4, 5, 200] | [1, 2] | [2, 200] | 1.827 | 0.381 | 4.80x | ✓ |
| 4D 多轴归约（小张量） | [2, 4, 5, 2] | [0, 2] | [4, 2] | 0.027 | 0.058 | 0.46x | ✓ |
| 4D 超大规模多轴归约 | [2, 4, 8, 1600000] | [2, 3] | [2, 4] | 22064.407 | 0.237 | 92984.98x | ✓ |
| 4D 大规模单轴归约 | [200, 4, 8, 1600] | [1] | [200, 8, 1600] | 2708.972 | 9.559 | 283.40x | ✓ |
| 4D 单轴归约（小张量） | [2, 4, 8, 8] | [1] | [2, 8, 8] | 0.143 | 0.161 | 0.89x | ✓ |
| 4D 单轴归约（小张量） | [2, 8, 8, 8] | [1] | [2, 8, 8] | 0.250 | 0.103 | 2.43x | ✓ |
| 4D 大规模单轴归约 | [2, 4, 800000, 16] | [2] | [2, 4, 16] | 21137.060 | 87.086 | 242.71x | ✓ |
| 4D 大规模多轴归约 | [2, 4, 8, 160000] | [1, 2] | [2, 160000] | 2330.285 | 9.906 | 235.24x | ✓ |
| 4D 多轴归约 | [2, 4, 800, 16] | [0, 2] | [4, 16] | 22.042 | 2.843 | 7.75x | ✓ |
| 8D 多轴归约 | [2, 4, 8, 3, 8, 28, 18, 26] | [0, 2, 4, 6] | [4, 3, 28, 26] | 4989.257 | 41.511 | 120.19x | ✓ |