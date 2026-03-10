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
 * \file reduce_all.cpp
 * \brief ReduceAll kernel entry point
 */

#include "reduce_all.h"
using namespace optiling;

template <uint32_t schMode>
__global__ __aicore__ void reduce_all(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReduceAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceAllTilingData, tilingData, tiling);

    NsReduceAll::KernelReduceAll op;
    op.Init(self, out, workspace, &tilingData);
    op.Process();
}