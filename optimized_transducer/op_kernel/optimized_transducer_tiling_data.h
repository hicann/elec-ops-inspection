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
 * \file optimized_transducer_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __OPTIMIZED_TRANSDUCER_TILING_DATA_H__
#define __OPTIMIZED_TRANSDUCER_TILING_DATA_H__

struct OptimizedTransducerTilingData {
    int64_t tileNum;
    int64_t blank;
    float clamp;
    bool fusedLogSoftmax;     
    int64_t totalPositions;   // 总位置数（点位数量），Σ(t,u)
    int64_t vocabSize;        // K，词表大小
    int64_t batchSize;         // N，批次大小
    int64_t maxTargetLength;   // targets 的最大长度（从 shape 获取）
    uint64_t ubSize;           // UB内存大小，单位：字节
    uint32_t blockSize;
    uint32_t usedCoreNum;      // 显式记录计划使用的核心数
};

#endif