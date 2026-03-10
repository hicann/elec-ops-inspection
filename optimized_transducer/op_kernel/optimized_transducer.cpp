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
 * \file optimized_transducer.cpp
 * \brief
 */

#include "optimized_transducer.h"

template <uint32_t schMode>
__global__ __aicore__ void optimized_transducer(GM_ADDR logits, GM_ADDR targets, GM_ADDR logitLengths,
                                                 GM_ADDR targetLengths, GM_ADDR loss, GM_ADDR grad,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(OptimizedTransducerTilingData);
    GET_TILING_DATA_WITH_STRUCT(OptimizedTransducerTilingData, tilingData, tiling);

    NsOptimizedTransducer::OptimizedTransducer op;
    op.Init(logits, targets, logitLengths, targetLengths, loss, grad, workspace, &tilingData);
    op.Process();
}