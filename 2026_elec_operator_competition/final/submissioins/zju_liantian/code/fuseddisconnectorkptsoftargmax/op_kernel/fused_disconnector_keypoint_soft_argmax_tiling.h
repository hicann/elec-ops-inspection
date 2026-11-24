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

#ifndef FUSED_DISCONNECTOR_KEYPOINT_SOFT_ARGMAX_TILING_H
#define FUSED_DISCONNECTOR_KEYPOINT_SOFT_ARGMAX_TILING_H

#include <cstdint>

struct FusedDisconnectorKeypointSoftArgmaxTilingData {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t tileRows;
    uint32_t groupsPerBatch;
    uint32_t groupFloats;
    uint32_t workspaceOffset;
    uint32_t parts;
    uint32_t useJoin;
    float invTau;
    float stride;
};

#endif  // FUSED_DISCONNECTOR_KEYPOINT_SOFT_ARGMAX_TILING_H
