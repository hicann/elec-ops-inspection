/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "unique_v3.h"
#include "unique_v3_counts.h"
#include "unique_v3_inverse.h"


extern "C" __global__ __aicore__ void unique_v3(
    GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt, 
    GM_ADDR inverse, GM_ADDR counts, 
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelUnique<DTYPE_INPUT> op(pipe);
    op.Init(input,
            output,
            uniqueCnt,
            inverse,
            counts,
            workspace,
            tiling_data.totalLength,
            tiling_data.shortBlockTileNum,
            tiling_data.tileLength,
            tiling_data.tailLength,
            tiling_data.aivNum,
            tiling_data.blockNum,
            tiling_data.shortBlockNum,
            tiling_data.flagInverse,
            tiling_data.flagCounts);
    op.Process();
}