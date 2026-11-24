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

#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"
#include "fdk_channel_banks.h"
#include "fdk_direct_projection.h"
#include "fdk_fallback.h"
#include "fdk_onchip_join.h"
#include "fdk_spatial_channels.h"
#include "fdk_uniform_output.h"
#include "fdk_vertical_cube.h"

template <uint32_t ROUTE, typename DT_FEAT>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, t, tiling);
    if constexpr (ROUTE == 0U) {
        fdk::RunFallback(feat, weight, bias, keypoints, scores, t);
    } else if constexpr (ROUTE == 1U) {
        fdk::RunUniformOutput(feat, weight, bias, keypoints, scores, t);
    } else if constexpr (ROUTE == 2U) {
        fdk::RunOnchipJoin<64, 4>(feat, weight, bias, keypoints, scores, t);
    } else if constexpr (ROUTE == 3U) {
        fdk::RunOnchipJoin<96, 2>(feat, weight, bias, keypoints, scores, t);
    } else if constexpr (ROUTE == 4U) {
        fdk::RunDirectProjection<8>(feat, weight, bias, keypoints, scores, t);
    } else if constexpr (ROUTE == 5U) {
        fdk::RunDirectProjection<16>(feat, weight, bias, keypoints, scores, t);
    } else if constexpr (ROUTE == 6U) {
        fdk::RunChannelBanks(feat, weight, bias, keypoints, scores, t);
    } else if constexpr (ROUTE == 7U) {
        fdk::RunPackedCube(feat, weight, bias, keypoints, scores, t);
    } else if constexpr (ROUTE == 8U) {
        fdk::RunHaloCube(feat, weight, bias, keypoints, scores, t);
    } else if constexpr (ROUTE == 9U) {
        fdk::RunSpatialChannels(feat, weight, bias, keypoints, scores, t);
    }
}
