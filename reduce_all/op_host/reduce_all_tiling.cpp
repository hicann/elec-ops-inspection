/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_all_tiling.cpp
 * \brief ReduceAll tiling function implementation
 *
 * 严格为 reduce_all.h kernel 服务的简化版：
 * 1) 轴融合：合并连续的 R/K 轴
 * 2) 计算 fused dims / fused axes / inputStrides
 * 3) 计算 outputStrides（按 fused 维展开，K 轴有效，R 轴填 0）
 * 4) tileSize & 按元素均匀分核
 * 5) 计算 workGmSize（每 core 一份，大小为 totalOutputSize 对齐到 64）
 */

#include <algorithm>
#include <vector>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_util.h"
#include "../op_kernel/reduce_all_tiling_data.h"
#include "../op_kernel/reduce_all_tiling_key.h"

namespace optiling {

struct ReduceAllCompileInfo {};

static inline uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static inline uint32_t AlignDown(uint32_t value, uint32_t alignment)
{
    return (value / alignment) * alignment;
}

static inline uint64_t AlignUp64(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

/**
 * @brief 从 dim 属性读取 axes 数据
 */
static ge::graphStatus GetReduceAxesFromAttr(
    gert::TilingContext* context,
    const gert::Shape& inputShape,
    std::vector<int64_t>& normalizedAxes)
{
    int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());
    normalizedAxes.clear();

    // 获取属性
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        // 属性不存在，默认全归约
        for (int64_t i = 0; i < rank; i++) {
            normalizedAxes.push_back(i);
        }
        return ge::GRAPH_SUCCESS;
    }

    // 获取 dim
    auto dimListPtr = attrs->GetListInt(0);
    if (dimListPtr == nullptr) {
        // dim 不存在，默认全归约
        for (int64_t i = 0; i < rank; i++) {
            normalizedAxes.push_back(i);
        }
        return ge::GRAPH_SUCCESS;
    }

    size_t axesNum = dimListPtr->GetSize();

    // dim 为空，默认全归约
    if (axesNum == 0) {
        for (int64_t i = 0; i < rank; i++) {
            normalizedAxes.push_back(i);
        }
        return ge::GRAPH_SUCCESS;
    }

    // 获取数据
    const int64_t* axesData = dimListPtr->GetData();
    if (axesData == nullptr) {
        // 数据为空，默认全归约
        for (int64_t i = 0; i < rank; i++) {
            normalizedAxes.push_back(i);
        }
        return ge::GRAPH_SUCCESS;
    }

    // 读 dim 属性中的各个轴
    normalizedAxes.reserve(axesNum);

    for (size_t i = 0; i < axesNum; ++i) {
        int64_t axis = axesData[i];

        if (axis < 0) {
            axis += rank;
        }
        OP_CHECK_IF(axis < 0 || axis >= rank,
            OP_LOGE(context, "[ReduceAll] axis %ld out of range [0, %ld)", axis, rank),
            return ge::GRAPH_FAILED);
        normalizedAxes.push_back(axis);
    }

    // 排序去重
    std::sort(normalizedAxes.begin(), normalizedAxes.end());
    normalizedAxes.erase(std::unique(normalizedAxes.begin(), normalizedAxes.end()), normalizedAxes.end());

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 解析并验证 axes；计算 totalInputSize；输出规范化 axes；标记空/全规约
 */
static bool ParseAndValidate(const gert::Shape& inputShape,
                             const std::vector<int64_t>& normalizedAxes,
                             uint64_t& totalInputSize,
                             bool& hasZeroDim,
                             bool& isFullReduce)
{
    int64_t ndim = static_cast<int64_t>(inputShape.GetDimNum());

    totalInputSize = 1;
    hasZeroDim = false;
    for (size_t i = 0; i < inputShape.GetDimNum(); ++i) {
        int64_t dimVal = inputShape.GetDim(i);
        if (dimVal == 0) {
            hasZeroDim = true;
        }
        totalInputSize *= static_cast<uint64_t>(dimVal);
    }

    isFullReduce = (normalizedAxes.size() == static_cast<size_t>(ndim));
    return true;
}

/**
 * @brief 轴融合：将连续的 R/K 轴合并
 */
static void FuseAxes(const gert::Shape& inputShape,
                     const std::vector<int64_t>& normalizedAxes,
                     std::vector<uint32_t>& fusedDims,
                     std::vector<uint32_t>& fusedIsReduce)
{
    size_t ndim = inputShape.GetDimNum();
    fusedDims.clear();
    fusedIsReduce.clear();

    if (ndim == 0) {
        return;
    }

    std::vector<bool> isReduceAxis(ndim, false);
    for (int64_t axis : normalizedAxes) {
        isReduceAxis[static_cast<size_t>(axis)] = true;
    }

    bool currentIsReduce = isReduceAxis[ndim - 1];
    uint64_t currentSize = static_cast<uint64_t>(inputShape.GetDim(ndim - 1));

    for (int64_t i = static_cast<int64_t>(ndim) - 2; i >= 0; --i) {
        size_t idx = static_cast<size_t>(i);
        if (isReduceAxis[idx] == currentIsReduce) {
            currentSize *= static_cast<uint64_t>(inputShape.GetDim(idx));
        } else {
            fusedDims.insert(fusedDims.begin(), static_cast<uint32_t>(currentSize));
            fusedIsReduce.insert(fusedIsReduce.begin(), currentIsReduce ? 1 : 0);

            currentIsReduce = isReduceAxis[idx];
            currentSize = static_cast<uint64_t>(inputShape.GetDim(idx));
        }
    }

    fusedDims.insert(fusedDims.begin(), static_cast<uint32_t>(currentSize));
    fusedIsReduce.insert(fusedIsReduce.begin(), currentIsReduce ? 1 : 0);
}

/**
 * @brief 计算 fused 维的 inputStrides
 */
static void ComputeFusedInputStrides(const std::vector<uint32_t>& fusedDims,
                                    std::vector<uint32_t>& inputStrides)
{
    inputStrides.clear();
    inputStrides.resize(fusedDims.size());

    uint64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(fusedDims.size()) - 1; i >= 0; --i) {
        inputStrides[static_cast<size_t>(i)] = static_cast<uint32_t>(stride);
        stride *= static_cast<uint64_t>(fusedDims[static_cast<size_t>(i)]);
    }
}

/**
 * @brief 计算 totalOutputSize + outputStrides
 */
static void ComputeOutputMetaByFused(const std::vector<uint32_t>& fusedDims,
                                    const std::vector<uint32_t>& fusedIsReduce,
                                    uint64_t& totalOutputSize,
                                    std::vector<uint32_t>& outputStridesByFused)
{
    totalOutputSize = 1;
    outputStridesByFused.clear();
    outputStridesByFused.resize(fusedDims.size(), 0);

    std::vector<uint32_t> kDims;
    std::vector<size_t> kPos;
    for (size_t i = 0; i < fusedDims.size(); ++i) {
        if (fusedIsReduce[i] == 0) {
            kDims.push_back(fusedDims[i]);
            kPos.push_back(i);
            totalOutputSize *= static_cast<uint64_t>(fusedDims[i]);
        }
    }

    if (kDims.empty()) {
        totalOutputSize = 1;  // 全规约：输出单元素
        return;
    }

    uint32_t stride = 1;
    for (int64_t i = static_cast<int64_t>(kDims.size()) - 1; i >= 0; --i) {
        size_t fusedIdx = kPos[static_cast<size_t>(i)];
        outputStridesByFused[fusedIdx] = stride;
        stride *= kDims[static_cast<size_t>(i)];
    }
}

/**
 * @brief 确定规约模式
 */
static uint32_t DetermineReduceMode(const std::vector<uint32_t>& fusedIsReduce,
                                   bool isFullReduce)
{
    if (isFullReduce) {
        return MODE_FULL_REDUCE;
    }

    size_t dimCount = fusedIsReduce.size();
    if (dimCount == 0) {
        return MODE_FULL_REDUCE;
    }

    if (dimCount == 2) {
        if (fusedIsReduce[0] == 0 && fusedIsReduce[1] == 1) {
            return MODE_KR;
        }
        if (fusedIsReduce[0] == 1 && fusedIsReduce[1] == 0) {
            return MODE_RK;
        }
    }

    if (dimCount == 3) {
        if (fusedIsReduce[0] == 1 && fusedIsReduce[1] == 0 && fusedIsReduce[2] == 1) {
            return MODE_RKR;
        }
        if (fusedIsReduce[0] == 0 && fusedIsReduce[1] == 1 && fusedIsReduce[2] == 0) {
            return MODE_KRK;
        }
    }

    return MODE_GENERAL;
}

static uint32_t ComputeMaxTileSize(uint64_t ubSize)
{
    uint32_t maxTileBytsSize = static_cast<uint32_t>((ubSize - FIXED_EXPENSES) / UB_BUFFER_FACTOR);
    maxTileBytsSize = AlignDown(maxTileBytsSize, BLOCK_SIZE);
    if (maxTileBytsSize < MIN_TILE_SIZE) {
        maxTileBytsSize = MIN_TILE_SIZE;
    }
    return maxTileBytsSize;
}

static void ComputeCoreAllocation(uint64_t totalElements,
                                  uint32_t tileSize,
                                  uint32_t availableCores,
                                  uint32_t& coreNum,
                                  uint64_t& elementsPerCore,
                                  uint32_t& largeCoreCount)
{
    if (totalElements == 0) {
        coreNum = 1;
        elementsPerCore = 0;
        largeCoreCount = 0;
        return;
    }

    uint64_t maxCores = (totalElements + tileSize - 1) / tileSize;
    coreNum = (maxCores < availableCores) ? static_cast<uint32_t>(maxCores) : availableCores;
    if (coreNum == 0) {
        coreNum = 1;
    }

    elementsPerCore = totalElements / coreNum;
    largeCoreCount = static_cast<uint32_t>(totalElements % coreNum);
}

static ge::graphStatus ReduceAllTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    const gert::StorageShape* selfShape = context->GetInputShape(0);
    OP_CHECK_IF(selfShape == nullptr, OP_LOGE(context, "selfShape is nullptr"), return ge::GRAPH_FAILED);
    const gert::Shape& inputShape = selfShape->GetStorageShape();

    // 从 dim 属性获取 axes
    std::vector<int64_t> normalizedAxes;
    ge::graphStatus axesStatus = GetReduceAxesFromAttr(context, inputShape, normalizedAxes);
    OP_CHECK_IF(axesStatus != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetReduceAxesFromAttr failed"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t availableCores = ascendcPlatform.GetCoreNum();

    ReduceAllTilingData* tiling = context->GetTilingData<ReduceAllTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ReduceAllTilingData), 0, sizeof(ReduceAllTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 解析输入参数
    uint64_t totalInputSize = 0;
    bool hasZeroDim = false;
    bool isFullReduce = false;

    if (!ParseAndValidate(inputShape, normalizedAxes, totalInputSize, hasZeroDim, isFullReduce)) {
        OP_LOGE(context, "ParseAndValidate failed");
        return ge::GRAPH_FAILED;
    }

    tiling->totalInputSize = totalInputSize;


    // 处理空输入
    if (hasZeroDim || totalInputSize == 0) {
        tiling->totalOutputSize = isFullReduce ? 1 : 0;
        tiling->workGmSize = AlignUp64(static_cast<uint64_t>(tiling->totalOutputSize), 64);

        tiling->coreNum = 1;
        tiling->ubSize = ubSize;
        tiling->tileSize = ComputeMaxTileSize(ubSize);
        tiling->elementsPerCore = 0;
        tiling->largeCoreCount = 0;
        tiling->reduceMode = MODE_FULL_REDUCE;

        context->SetBlockDim(tiling->coreNum);
        context->SetTilingKey(GET_TPL_TILING_KEY(REDUCE_ALL_KEY_BOOL));

        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);

        // 为 kernel 预留：sys + per-core workGm
        size_t workGmSize = static_cast<size_t>(tiling->coreNum) * tiling->workGmSize;
        currentWorkspace[0] = sysWorkspaceSize + workGmSize + 64;
        return ge::GRAPH_SUCCESS;
    }

    // 轴融合
    std::vector<uint32_t> fusedDims;
    std::vector<uint32_t> fusedIsReduce;
    FuseAxes(inputShape, normalizedAxes, fusedDims, fusedIsReduce);

    tiling->fusedDimCount = static_cast<uint32_t>(fusedDims.size());
    OP_CHECK_IF(tiling->fusedDimCount > MAX_DIMS, OP_LOGE(context, "fusedDimCount overflow"), return ge::GRAPH_FAILED);
    
    for (size_t i = 0; i < fusedDims.size(); ++i) {
        tiling->fusedDims[i] = fusedDims[i];
        tiling->fusedAxes[i] = fusedIsReduce[i];
    }

    // fused input strides（kernel General 必须）
    std::vector<uint32_t> inputStrides;
    ComputeFusedInputStrides(fusedDims, inputStrides);
    for (size_t i = 0; i < inputStrides.size(); ++i) {
        tiling->inputStrides[i] = inputStrides[i];
    }

    // output meta（totalOutputSize + outputStridesByFused）
    uint64_t totalOutputSize = 0;
    std::vector<uint32_t> outputStridesByFused;
    ComputeOutputMetaByFused(fusedDims, fusedIsReduce, totalOutputSize, outputStridesByFused);

    tiling->totalOutputSize = totalOutputSize;
    for (size_t i = 0; i < outputStridesByFused.size(); ++i) {
        tiling->outputStrides[i] = outputStridesByFused[i];
    }

    // 模式选择
    tiling->reduceMode = DetermineReduceMode(fusedIsReduce, isFullReduce);
    
    // ubSize
    tiling->ubSize = ubSize;

    // tileSize
    tiling->tileSize = ComputeMaxTileSize(ubSize);

    // 核间分配
    ComputeCoreAllocation(totalInputSize, tiling->tileSize, availableCores,
                          tiling->coreNum, tiling->elementsPerCore, tiling->largeCoreCount);

    // workGM
    tiling->workGmSize = AlignUp64(totalOutputSize, 64);

    context->SetBlockDim(tiling->coreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(REDUCE_ALL_KEY_BOOL));

    // workspace：sys + 每 core 一份 workGm
    size_t usrSize = 0;
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize + static_cast<size_t>(tiling->coreNum) * tiling->workGmSize;


    // 在这里 直接使用 printf 打印所有的 tilingdata 
    // printf("totalInputSize: %lu\n", tiling->totalInputSize);
    // printf("totalOutputSize: %lu\n", tiling->totalOutputSize);
    // printf("workGmSize: %lu\n", tiling->workGmSize);
    // printf("fusedDimCount: %u\n", tiling->fusedDimCount);
    // printf("reduceMode: %u\n", tiling->reduceMode);
    // printf("tileSize: %u\n", tiling->tileSize);
    // printf("coreNum: %u\n", tiling->coreNum);
    // printf("elementsPerCore: %lu\n", tiling->elementsPerCore);
    // printf("largeCoreCount: %u\n", tiling->largeCoreCount);
    // for (size_t i = 0; i < fusedDims.size(); ++i) {
    //     printf("fusedDims[%lu]: %u\n", i, tiling->fusedDims[i]);
    //     printf("fusedAxes[%lu]: %u\n", i, tiling->fusedAxes[i]);
    // }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReduceAll([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReduceAll).Tiling(ReduceAllTilingFunc).TilingParse<ReduceAllCompileInfo>(TilingParseForReduceAll);

}  // namespace optiling