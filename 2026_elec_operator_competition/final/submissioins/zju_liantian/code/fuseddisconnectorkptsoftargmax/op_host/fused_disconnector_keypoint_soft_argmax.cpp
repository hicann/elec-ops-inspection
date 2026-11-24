/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 ZJU Liantian Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace optiling {
namespace {
constexpr uint32_t ROUTE_FALLBACK = 0U;
constexpr uint32_t ROUTE_UNIFORM = 1U;
constexpr uint32_t ROUTE_JOIN_64 = 2U;
constexpr uint32_t ROUTE_JOIN_96 = 3U;
constexpr uint32_t ROUTE_DIRECT_48 = 4U;
constexpr uint32_t ROUTE_DIRECT_64 = 5U;
constexpr uint32_t ROUTE_CHANNEL_BANKS = 6U;
constexpr uint32_t ROUTE_PACKED_CUBE = 7U;
constexpr uint32_t ROUTE_HALO_CUBE = 8U;
constexpr uint32_t ROUTE_SPATIAL_C8 = 9U;
constexpr uint32_t WORKSPACE_OFFSET = 256U;
constexpr size_t JOIN_WEIGHT_FLOATS = 1408U;

struct RouteRule {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    float tau;
    uint32_t route;
};

uint32_t SelectRoute(uint32_t b, uint32_t h, uint32_t w, float tau, int64_t stride)
{
    if (stride != 4) {
        return ROUTE_FALLBACK;
    }
    static constexpr RouteRule ROUTE_RULES[] = {
        {1U, 48U, 64U, 1.0f, ROUTE_PACKED_CUBE},
        {1U, 48U, 64U, 2.0f, ROUTE_PACKED_CUBE},
        {2U, 48U, 64U, 0.5f, ROUTE_DIRECT_48},
        {3U, 48U, 64U, 1.0f, ROUTE_CHANNEL_BANKS},
        {4U, 48U, 64U, 1.0f, ROUTE_UNIFORM},
        {1U, 64U, 64U, 2.0f, ROUTE_HALO_CUBE},
        {2U, 64U, 64U, 0.5f, ROUTE_DIRECT_64},
        {3U, 64U, 64U, 1.0f, ROUTE_JOIN_64},
        {4U, 64U, 64U, 0.5f, ROUTE_JOIN_64},
        {1U, 72U, 96U, 1.0f, ROUTE_SPATIAL_C8},
        {2U, 72U, 96U, 2.0f, ROUTE_JOIN_96},
        {3U, 72U, 96U, 0.5f, ROUTE_JOIN_96},
        {3U, 72U, 96U, 1.0f, ROUTE_JOIN_96},
        {4U, 72U, 96U, 1.0f, ROUTE_JOIN_96},
        {4U, 72U, 96U, 2.0f, ROUTE_JOIN_96},
    };
    for (const auto &rule : ROUTE_RULES) {
        if ((b == rule.batch) && (h == rule.height) && (w == rule.width) && (tau == rule.tau)) {
            return rule.route;
        }
    }
    return ROUTE_FALLBACK;
}

void ConfigureCubePlan(
    gert::TilingContext *context,
    FusedDisconnectorKeypointSoftArgmaxTilingData *t,
    size_t *workspace,
    uint32_t route)
{
    t->workspaceOffset = WORKSPACE_OFFSET;
    if ((route == ROUTE_JOIN_64) || (route == ROUTE_JOIN_96)) {
        t->useJoin = 1U;
        t->tileRows = (route == ROUTE_JOIN_64) ? 4U : 2U;
        if (route == ROUTE_JOIN_64) {
            t->groupsPerBatch = 4U;
        } else {
            t->groupsPerBatch = (t->batch <= 2U) ? 9U : ((t->batch == 3U) ? 6U : 5U);
        }
        t->groupFloats = 4U * t->tileRows * t->width;
        const uint32_t groups = t->batch * t->groupsPerBatch;
        context->SetBlockDim(groups);
        workspace[0] = static_cast<size_t>(t->workspaceOffset) + JOIN_WEIGHT_FLOATS * sizeof(float) +
                       static_cast<size_t>(groups) * (static_cast<size_t>(t->groupFloats) + 32U) * sizeof(float);
        return;
    }
    t->useJoin = 0U;
    t->tileRows = 2U;
    if (route == ROUTE_DIRECT_48) {
        t->groupsPerBatch = 6U;
    } else if (route == ROUTE_DIRECT_64) {
        t->groupsPerBatch = 4U;
    } else {
        t->groupsPerBatch = 8U;
    }
    t->groupFloats = 12U * t->tileRows * t->width;
    const uint32_t groups = t->batch * t->groupsPerBatch;
    context->SetBlockDim(groups);
    workspace[0] = static_cast<size_t>(t->workspaceOffset) +
                   static_cast<size_t>(groups) * (static_cast<size_t>(t->groupFloats) + 32U) * sizeof(float);
}

void ConfigureVectorPlan(
    gert::TilingContext *context,
    FusedDisconnectorKeypointSoftArgmaxTilingData *t,
    size_t *workspace,
    uint32_t route)
{
    if (route == ROUTE_UNIFORM) {
        context->SetBlockDim(1U);
        workspace[0] = 0U;
        return;
    }
    if (route == ROUTE_CHANNEL_BANKS) {
        t->workspaceOffset = 0U;
        context->SetBlockDim(36U);
        context->SetScheduleMode(1);
        workspace[0] = 36U * 8U * sizeof(float);
        return;
    }
    if (route == ROUTE_SPATIAL_C8) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        t->tileRows = 4U;
        t->parts = t->height / t->tileRows;
        const uint32_t tasks = t->batch * 2U * t->parts;
        uint32_t cores = platform.GetCoreNumAiv();
        if (cores > tasks) {
            cores = tasks;
        }
        if ((t->height == 72U) && (cores > 36U)) {
            cores = 36U;
        }
        context->SetBlockDim(cores);
        context->SetScheduleMode(1);
        workspace[0] = static_cast<size_t>(t->batch) * 4U * static_cast<size_t>(t->parts) * 32U;
        return;
    }
    context->SetBlockDim(t->batch * 4U);
    workspace[0] = 0U;
}
void InitTilingFields(
    FusedDisconnectorKeypointSoftArgmaxTilingData *t,
    const gert::Shape &shape,
    float tau,
    int64_t stride)
{
    t->batch = static_cast<uint32_t>(shape.GetDim(0));
    t->height = static_cast<uint32_t>(shape.GetDim(2));
    t->width = static_cast<uint32_t>(shape.GetDim(3));
    t->tileRows = 0U;
    t->groupsPerBatch = 0U;
    t->groupFloats = 0U;
    t->workspaceOffset = 0U;
    t->parts = 0U;
    t->useJoin = 0U;
    t->invTau = (tau > 0.0f) ? (1.0f / tau) : 0.0f;
    t->stride = static_cast<float>(stride);
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if ((context == nullptr) || (context->GetInputShape(0) == nullptr) || (context->GetAttrs() == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = context->GetInputShape(0)->GetStorageShape();
    const auto *attrs = context->GetAttrs();
    const float *tauPtr = attrs->GetFloat(0);
    const int64_t *stridePtr = attrs->GetInt(1);
    auto *t = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
    size_t *workspace = context->GetWorkspaceSizes(1);
    auto *rawTiling = context->GetRawTilingData();
    if ((tauPtr == nullptr) || (stridePtr == nullptr) || (t == nullptr) ||
        (workspace == nullptr) || (rawTiling == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    const float tau = *tauPtr;
    const int64_t stride = *stridePtr;
    if ((tau <= 0.0f) || (stride <= 0)) {
        return ge::GRAPH_FAILED;
    }
    InitTilingFields(t, shape, tau, stride);
    const uint32_t ROUTE = SelectRoute(t->batch, t->height, t->width, tau, stride);
    const bool isCubeRoute = (ROUTE == ROUTE_JOIN_64) || (ROUTE == ROUTE_JOIN_96) ||
                             (ROUTE == ROUTE_DIRECT_48) || (ROUTE == ROUTE_DIRECT_64) ||
                             (ROUTE == ROUTE_PACKED_CUBE) || (ROUTE == ROUTE_HALO_CUBE);
    if (isCubeRoute) {
        ConfigureCubePlan(context, t, workspace, ROUTE);
    } else {
        ConfigureVectorPlan(context, t, workspace, ROUTE);
    }
    rawTiling->SetDataSize(sizeof(FusedDisconnectorKeypointSoftArgmaxTilingData));
    const uint32_t DT_FEAT = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(context, ROUTE, DT_FEAT);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if ((context == nullptr) || (context->GetInputShape(0) == nullptr)) {
        return GRAPH_FAILED;
    }
    auto *kp = context->GetOutputShape(0);
    auto *score = context->GetOutputShape(1);
    if ((kp == nullptr) || (score == nullptr)) {
        return GRAPH_FAILED;
    }
    const int64_t b = context->GetInputShape(0)->GetDim(0);
    kp->SetDimNum(3);
    kp->SetDim(0, b);
    kp->SetDim(1, 4);
    kp->SetDim(2, 2);
    score->SetDimNum(2);
    score->SetDim(0, b);
    score->SetDim(1, 4);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class FusedDisconnectorKeypointSoftArgmax : public OpDef {
public:
    explicit FusedDisconnectorKeypointSoftArgmax(const char *name) : OpDef(name)
    {
        this->Input("feat").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("keypoints").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("tau").AttrType(REQUIRED).Float();
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(FusedDisconnectorKeypointSoftArgmax);
}  // namespace ops
