// Copyright 2026 Electrical Engineering SIG - CANN Community
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace optiling {

constexpr uint32_t INDEX_ZERO = 0;
constexpr uint32_t INDEX_ONE = 1;
constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t INDEX_THREE = 3;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

struct OptimizedTransducerCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取shape、属性信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalLength, int64_t& vocabSize,
                                          int64_t& batchSize, int64_t& maxTargetLength, int64_t& blank, float& clamp,
                                          bool& fusedLogSoftmax)
{
    // 获取输入shape信息
    auto logitsShape = context->GetInputShape(INDEX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitsShape);
    auto storageShape = logitsShape->GetStorageShape();
    totalLength = storageShape.GetShapeSize();
    
    // logits shape: [Σ(t,u), V]
    if (storageShape.GetDimNum() == 2) {
        vocabSize = storageShape.GetDim(1);
    } else {
        OP_LOGE(context, "logits shape should be 2D [Σ(t,u), V]");
        return ge::GRAPH_FAILED;
    }
    
    // 获取batchSize（从logitLengths）
    auto logitLengthsShape = context->GetInputShape(INDEX_TWO);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitLengthsShape);
    auto logitLengthsStorageShape = logitLengthsShape->GetStorageShape();
    if (logitLengthsStorageShape.GetDimNum() == 1) {
        batchSize = logitLengthsStorageShape.GetDim(0);
    } else {
        OP_LOGE(context, "logitLengths shape should be 1D [B]");
        return ge::GRAPH_FAILED;
    }

    // 获取targets shape: [B, maxTargetLen]
    auto targetsShape = context->GetInputShape(INDEX_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetsShape);
    auto targetsStorageShape = targetsShape->GetStorageShape();
    maxTargetLength = targetsStorageShape.GetDim(1);

    // dtype校验 - 只支持 float32
    auto inputDesc = context->GetInputDesc(INDEX_ZERO);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    if (dataType != ge::DT_FLOAT) {
        OP_LOGE(context, "only support DT_FLOAT dtype");
        return ge::GRAPH_FAILED;
    }

    // 获取属性值
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* blankPtr = attrs->GetInt(INDEX_ZERO);
    blank = blankPtr ? *blankPtr : -1;

    const float* clampPtr = attrs->GetFloat(INDEX_ONE);
    clamp = clampPtr ? *clampPtr : -1.0f;

    const bool* fusedPtr = attrs->GetBool(INDEX_TWO);
    fusedLogSoftmax = fusedPtr ? *fusedPtr : true;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, int64_t totalPositions)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    size_t alphaBetaSize = 2 * totalPositions * sizeof(float);
    currentWorkspace[0] = sysWorkspaceSize + alphaBetaSize;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus OptimizedTransducerTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取shape、属性信息
    int64_t totalLength;
    int64_t vocabSize;
    int64_t batchSize;
    int64_t maxTargetLength;
    int64_t blank;
    float clamp;
    bool fusedLogSoftmax;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalLength, vocabSize, batchSize, maxTargetLength, blank, clamp, fusedLogSoftmax) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. 获取WorkspaceSize信息
    int64_t totalPositions = totalLength / vocabSize;
    OP_CHECK_IF(
        GetWorkspaceSize(context, totalPositions) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. 设置tiling信息
    OptimizedTransducerTilingData* tiling = context->GetTilingData<OptimizedTransducerTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(OptimizedTransducerTilingData), 0, sizeof(OptimizedTransducerTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalPositions = totalLength / vocabSize;
    tiling->blank = blank;
    tiling->clamp = clamp;
    tiling->fusedLogSoftmax = fusedLogSoftmax;
    tiling->vocabSize = vocabSize;
    tiling->batchSize = batchSize;
    tiling->maxTargetLength = maxTargetLength;
    tiling->ubSize = ubSize;
    tiling->blockSize = Ops::Base::GetUbBlockSize(context);

    // 设置核心数
    uint32_t usedCoreNum = coreNum;
    // uint32_t usedCoreNum = 1;
    if (batchSize > 0 && batchSize < usedCoreNum) {
        usedCoreNum = batchSize;
    }
    tiling->usedCoreNum = usedCoreNum;
    context->SetBlockDim(usedCoreNum);

    // 5. 设置tiling key
    uint64_t tilingKey = GET_TPL_TILING_KEY(TRANSDUCER_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForOptimizedTransducer([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(OptimizedTransducer)
    .Tiling(OptimizedTransducerTilingFunc)
    .TilingParse<OptimizedTransducerCompileInfo>(TilingParseForOptimizedTransducer);

} // namespace optiling