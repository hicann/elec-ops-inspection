# ReduceAll 算子设计文档

## 1. 目标

将 TBE 实现的 ReduceAll 算子用 Ascend C 重新实现，并提交到算子开源仓。

- 支持 int8 (bool)
- 性能不低于 TBE 算子的 95%
- TBE kernel 实现：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/
- TBE 算子原型定义：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/
- TBE 算子信息库（910B 配置）：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b

## 2. 算子概述

ReduceAll 算子沿指定轴对布尔张量执行逻辑与（AND）归约操作。当指定轴上所有元素均为 True 时，输出 True；否则输出 False。

设输入张量为 $self$，归约轴为 $axes$，则输出 $out$ 的计算公式为：

$$out = \bigwedge_{axes} self$$

其中 $\bigwedge$ 表示逻辑与操作。

## 3. TBE 的实现

根据算子信息库，TBE 算子支持的数据类型和数据格式如下：

| 参数 | 名称 | 数据类型 | 数据格式 | 参数类型 |
|------|------|----------|----------|----------|
| input0 | x | bool | ND | required |
| input1 | axes | int32, int64 | ND | required |
| output0 | y | bool | ND | required |
| attr | keep_dims | bool | - | optional (默认值: false) |

TBE 的 reduce_all 的计算流程：
1. 参数预处理：处理 keep_dims 默认值，检查输入数据类型（bool 转为 int8），检查 axes 数据类型（int32 或 int64）
2. 设置关系位置标记，记录原始 shape，规约后的 shape，需要规约的轴
3. 调用 classify 进行输入分类，获取不同的输入组合
4. 对每个输入组合，创建占位符，调用 reduce_all_compute 计算，生成调度并构建
5. 在 reduce_all_compute 计算中，先判断输入张量如果存在零维度（zero_tensor_flg = 0 in x.shape），走快速路径
6. 否则采用等价的数值计算方式开始计算
   - 将 int8 (bool) 转换为 float16，调用 tbe.cast_to(x, "float16")
   - 调用 tbe.vabs(data_fp16)，确保 True=1 和 False=0 都为非负值
   - 沿指定轴调用 tbe.reduce_min(data_abs, axis=axes, keepdims=keepdims)求该轴的最小值
   - 调用 tbe.cast_to(res_any, dtype)将结果转回 int8

## 4. Ascend C 算子设计

### 4.1 算子原型

Ascend C 实现的算子原型设计如下，与 TBE 算子原型保持一致：

| 参数类型 | 名称 | 数据类型 | 格式 | 说明 |
|----------|------|----------|------|------|
| 输入 | self | int8 (bool) | ND | 布尔张量 |
| 输入 | axes | int32, int64 | ND | 归约轴 |
| 输出 | out | int8 (bool) | ND | 归约结果 |
| 属性 | keepdim | bool | - | 是否保留归约维度，默认 false |

### 4.2 算法设计

#### 4.2.1 计算路径

1. 当输入张量 shape 中包含 0 时，直接输出 True（对空集合的与操作返回 True）
2. 否则走数值模拟计算路径，核心计算步骤为：
   - CopyIn：根据坐标计算将数据从 Global Memory 搬运到 Local Memory
   - Cast：将 int8 数据转换为 float16
   - Abs：对 float16 数据取绝对值（确保 True 的任意非零表示都转为正数）
   - ReduceMin：求最小值
   - Cast：将 float16 结果转换回 int8
   - CopyOut：将结果从 Local Memory 写回 Global Memory

#### 4.2.2 核心算法

聚合操作将原来的多个元素归约为一个元素，例如 100 个元素每 10 个求和后得到 10 个元素，shape 和轴的数量都变了。

核心要素是新旧坐标的转换，使用坐标和步长 strides 来计算偏移量。由于聚合操作需要将元素的步长分为两段：`base_offset` 和 `relative_offset`。

因为聚合意味着一个输出元素对应一组输入元素，组长就是 `total_reduce_elements`：

```cpp
// 计算给定形状的连续内存步长
std::vector<size_t> compute_strides(const std::vector<int>& shape) {
    if (shape.empty()) return {};
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// 多维坐标转一维偏移：根据多维坐标和步长计算一维内存偏移量
size_t coords_to_offset(const std::vector<int>& coords, const std::vector<size_t>& strides) {
    size_t offset = 0;
    for (size_t i = 0; i < coords.size(); ++i) {
        offset += coords[i] * strides[i];
    }
    return offset;
}

// 一维索引转多维坐标：将线性索引转换为多维坐标
std::vector<int> index_to_coords(size_t index, const std::vector<int>& shape) {
    int ndim = static_cast<int>(shape.size());
    std::vector<int> coords(ndim);
    for (int d = ndim - 1; d >= 0; --d) {
        coords[d] = index % shape[d];
        index /= shape[d];
    }
    return coords;
}
```

**准备阶段**：分离保留轴和聚合轴的 shape 和 strides

```cpp
// 计算原始数据的完整步长
std::vector<size_t> in_strides = compute_strides(in_shape);

// vector 转 set 方便后面 in 判断
std::set<int> reduce_set(reduce_axes.begin(), reduce_axes.end());

std::vector<int> kept_shape;          // 保留轴的 shape
std::vector<size_t> kept_strides;     // 保留轴的 strides

std::vector<int> reduced_shape;       // 聚合轴的 shape
std::vector<size_t> reduced_strides;  // 聚合轴的 strides

// 分离保留轴和聚合轴
for (int i = 0; i < ndim; ++i) {
    if (reduce_set.count(i)) {
        reduced_shape.push_back(in_shape[i]);
        reduced_strides.push_back(in_strides[i]);
    } else {
        kept_shape.push_back(in_shape[i]);
        kept_strides.push_back(in_strides[i]);
    }
}

// 计算所有输出元素的个数
size_t total_output_elements = 1;
for (int s : kept_shape) total_output_elements *= s;

// 计算每次聚合的元素的个数
size_t total_reduce_elements = 1;
for (int s : reduced_shape) total_reduce_elements *= s;
```

**核心逻辑**：

```cpp
// 外层循环：逐个遍历每个输出的元素，这就是一次聚合
for (size_t i = 0; i < total_output_elements; ++i) {
    
    // 计算第 i 个输出元素的保留轴坐标 (index_to_coords)
    std::vector<int> kept_coords = index_to_coords(i, kept_shape);

    // 计算 base_offset (coords_to_offset)
    // 聚合出第 i 个输出元素的那些原始元素是一个不连续组，该组的起始地址记为 base_offset
    size_t base_offset = coords_to_offset(kept_coords, kept_strides);

    float min_val = FLT_MAX;

    // 内层循环：遍历本次聚合的每个元素
    for (size_t j = 0; j < total_reduce_elements; ++j) {
        
        // 计算聚合轴坐标 (index_to_coords)
        std::vector<int> reduced_coords = index_to_coords(j, reduced_shape);
        
        // 计算 relative_offset (coords_to_offset)
        size_t relative_offset = coords_to_offset(reduced_coords, reduced_strides);

        // 使用 base_offset + relative_offset 访问原始数据
        float val = abs(input_data[base_offset + relative_offset]);
        if (val < min_val) min_val = val;
    }

    output_data[i] = (int8_t)min_val;
}
```

### 4.3 Tiling 策略

#### 4.3.1 多核划分

按照输出元素数量进行多核划分：
- `total_output_elements` = 所有保留轴维度的乘积
- 均匀分配: `coreOutputNum` = `total_output_elements` / `coreNum`
- 尾部处理: 余数分配给前面几个核心

#### 4.3.2 Tile 大小计算

UB 空间分配：
- 输入 buffer: `tileSize` × sizeof(int8) × 2 (Ping-Pong)
- 计算 buffer: `tileSize` × sizeof(float16) × 2 (数据 + reduce 工作区)
- 输出 buffer: `tileBatchNum` × sizeof(int8) × 2 (Ping-Pong)

计算 tileCapacity：
```
bytesPerElement = sizeof(int8) × 2 + sizeof(float16) × 2 = 6 bytes
tileCapacity = availableUBSize / bytesPerElement
```

#### 4.3.3 统一计算模式

采用统一的 Batch 计算逻辑：

1. **正常情况 (`total_reduce_elements` <= `tileCapacity`)**：
   - `tileBatchNum` = `tileCapacity` / `total_reduce_elements`
   - `tileSize` = `tileBatchNum` × `total_reduce_elements`
   - 一个 tile 处理多个完整聚合

2. **大 reduce 情况 (`total_reduce_elements` > `tileCapacity`)**：
   - `tileBatchNum` = 1
   - `tileSize` = `tileCapacity`
   - 多个 tile 处理一个聚合
   - 维护跨 tile 的 `partialMin` 状态

### 4.4 UB 布局设计

- 输入 buffer：使用 TQue 开启 double buffer，每块存放一个 tile 的 int8 输入数据
- 输出 buffer：使用 TQue 开启 double buffer，每块存放一个 tile 的 int8 输出数据
- 计算 buffer：不参与流水，用于向量计算与归约操作

## 5. TilingData 结构

```cpp
struct ReduceAllTilingData {
    // 基础形状参数
    int64_t totalOutputElements;    // 输出元素总数
    int64_t reduceElements;    // 每次聚合的元素数

    // 多核划分参数
    int64_t coreOutputNum;          // 每核处理的输出数
    int64_t tailCoreNum;            // 余数核数

    // Tile 分块参数
    int64_t tileSize;               // 每个 tile 的元素数
    int64_t tileNum;                // tile 数量
    int64_t tailTileSize;           // 尾 tile 大小
    int64_t tileBatchNum;           // 每个 tile 处理的完整聚合数

    // 保留轴参数（用于计算 base_offset）
    int32_t keptAxisNum;                    // 保留轴数量
    int64_t keptShape[MAX_DIM_NUM];         // 保留轴形状
    int64_t keptStrides[MAX_DIM_NUM];       // 保留轴步长

    // 聚合轴参数（用于计算 relative_offset）
    int32_t reducedAxisNum;                 // 聚合轴数量
    int64_t reducedShape[MAX_DIM_NUM];      // 聚合轴形状
    int64_t reducedStrides[MAX_DIM_NUM];    // 聚合轴步长
};
```

## 6. 算子约束限制

1. 数据类型约束：输入输出仅支持 int8 (bool) 类型
2. axes 类型约束：归约轴索引支持 int32 和 int64 类型
3. 维度约束：支持最大 8D 的输入张量
4. 归约轴约束：支持任意轴组合的归约，包括不连续轴
5. 空张量约束：支持包含零维度的张量输入（返回 True）
