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
 * \file reduce_all.h
 * \brief ReduceAll kernel implementation
 */

#ifndef REDUCE_ALL_H_
#define REDUCE_ALL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_all_tiling_data.h"
#include "reduce_all_tiling_key.h"

namespace NsReduceAll {

using namespace AscendC;
using namespace optiling;

class KernelReduceAll {
public:
    __aicore__ inline KernelReduceAll() {}

    __aicore__ inline void Init(GM_ADDR selfGM, GM_ADDR outGM, GM_ADDR workspace,
                                const ReduceAllTilingData* tilingData);

    __aicore__ inline void Process();

private:
    const ReduceAllTilingData* tiling_;

    GlobalTensor<int8_t> inputGM_;
    GlobalTensor<int8_t> outputGM_;
    GlobalTensor<int8_t> workGm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    TBuf<QuePosition::VECCALC> workBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> minBuf_;

    uint32_t coreId_;
    uint32_t coreNum_;
    uint64_t myStart_;
    uint64_t myCount_;
    uint32_t ubSize_;
    uint32_t tileSize_;
    uint64_t workGmSize_;
    uint64_t totalOutputSize_;
    uint64_t totalInputSize_;
    uint64_t alignedWorkspace_;

    __aicore__ inline void InitOutputAndWorkGm(GM_ADDR workGM, GM_ADDR outGM);
    __aicore__ inline void ComputeCoreRange();

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t alignment);
    __aicore__ inline uint32_t AlignDown(uint32_t value, uint32_t alignment);
    __aicore__ inline uint32_t GlobalIdxToOutputIdx(uint64_t globalIdx);

    __aicore__ inline void TileCopyIn(uint64_t gmOffset, uint32_t copySize);
    __aicore__ inline void CopyWorkGmToOutput();

    __aicore__ inline void ProcessAllReduce();
    __aicore__ inline void ProcessRK();
    __aicore__ inline void ProcessKR();
    __aicore__ inline void ProcessRKR();
    __aicore__ inline void ProcessKRK();
    __aicore__ inline void ProcessGeneral();
};

__aicore__ inline void KernelReduceAll::Init(GM_ADDR selfGM, GM_ADDR outGM, GM_ADDR workspace,
                                              const ReduceAllTilingData* tilingData) {


    tiling_ = tilingData;
    ubSize_ = tiling_->ubSize;
    coreId_ = GetBlockIdx();
    coreNum_ = tiling_->coreNum;
    tileSize_ = tiling_->tileSize;
    workGmSize_ = tiling_->workGmSize;
    totalInputSize_  = tiling_->totalInputSize;
    totalOutputSize_ = tiling_->totalOutputSize;
    alignedWorkspace_ = ((reinterpret_cast<uint64_t>(workspace) + 63) / 64) * 64;
    ComputeCoreRange();

    uint64_t workGmAddrU64 = alignedWorkspace_ + coreId_ * workGmSize_;
    GM_ADDR workGmAddr = reinterpret_cast<GM_ADDR>(workGmAddrU64);
    
    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(selfGM), totalInputSize_);
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(outGM), totalOutputSize_);
    workGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(workGmAddr), workGmSize_);

    inputGM_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    outputGM_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

    InitOutputAndWorkGm(workGmAddr, outGM);

    pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileSize_ * sizeof(int8_t));   // 2
    pipe_.InitBuffer(workBuf_, tileSize_  * sizeof(half));                   // 2 
    pipe_.InitBuffer(tmpBuf_, tileSize_  * sizeof(half));                    // 2
    pipe_.InitBuffer(minBuf_, 32 * sizeof(half));

}

__aicore__ inline void KernelReduceAll::Process() {

    if (myCount_ == 0) {
        return;
    }

    switch (tiling_->reduceMode) {
        case MODE_FULL_REDUCE:
            ProcessAllReduce();
            break;
        case MODE_KR:
            ProcessKR();
            break;
        case MODE_RK:
            ProcessRK();
            break;
        case MODE_RKR:
            ProcessRKR();
            break;
        case MODE_KRK:
            ProcessKRK();
            break;
        case MODE_GENERAL:
        default:
            ProcessGeneral();
            break;
    }
    // printf("Core %u finished processing.\n", coreId_);
    
    // DataCacheCleanAndInvalid<int8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workGm_);

    CopyWorkGmToOutput();
}


__aicore__ inline void KernelReduceAll::InitOutputAndWorkGm(GM_ADDR workGM, GM_ADDR outGM) {
    // core0 初始化 outputGM_ 为全1
    if (coreId_ == 0) {
        uint64_t totalInt16Count = totalOutputSize_ / 2;
        if (totalInt16Count > 0) {
            GlobalTensor<int16_t> outputGMInt16;
            outputGMInt16.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(outGM), totalInt16Count);
            Fill(outputGMInt16, totalInt16Count, static_cast<int16_t>(0x0101));

            if (totalOutputSize_ & 1) {
                outputGM_.SetValue(totalOutputSize_ - 1, static_cast<int8_t>(1));
            }
        } else if (totalOutputSize_ > 0) {
            outputGM_.SetValue(0, static_cast<int8_t>(1));
        }
        DataCacheCleanAndInvalid<int8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(outputGM_);
    }

    // 每个核心初始化自己的 workGm_ 为全1
    uint64_t workInt16Count = workGmSize_ / 2;
    if (workInt16Count > 0) {
        GlobalTensor<int16_t> workGmInt16;
        workGmInt16.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(workGM), workInt16Count);
        Fill(workGmInt16, workInt16Count, static_cast<int16_t>(0x0101));

        if (workGmSize_ & 1) {
            workGm_.SetValue(workGmSize_ - 1, static_cast<int8_t>(1));
        }
    }
    
    SyncAll();
}

__aicore__ inline void KernelReduceAll::ComputeCoreRange() {
    if (coreId_ >= coreNum_) {
        myStart_ = 0;
        myCount_ = 0;
        return;
    }

    uint64_t elementsPerCore = tiling_->elementsPerCore;
    uint32_t largeCoreCount = tiling_->largeCoreCount;

    if (coreId_ < largeCoreCount) {
        myStart_ = coreId_ * (elementsPerCore + 1);
        myCount_ = elementsPerCore + 1;
    } else {
        myStart_ = largeCoreCount * (elementsPerCore + 1) + (coreId_ - largeCoreCount) * elementsPerCore;
        myCount_ = elementsPerCore;
    }
}

__aicore__ inline uint32_t KernelReduceAll::AlignUp(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__aicore__ inline uint32_t KernelReduceAll::AlignDown(uint32_t value, uint32_t alignment) {
    return (value / alignment) * alignment;
}

__aicore__ inline uint32_t KernelReduceAll::GlobalIdxToOutputIdx(uint64_t globalIdx) {
    uint32_t fusedDimCount = tiling_->fusedDimCount;
    uint32_t outputIdx = 0;
    uint64_t remaining = globalIdx;
    
    for (uint32_t i = 0; i < fusedDimCount; ++i) {
        uint32_t stride = tiling_->inputStrides[i];
        uint32_t coord = static_cast<uint32_t>(remaining / stride);
        remaining = remaining % stride;
        
        if (tiling_->fusedAxes[i] == 0) {  // K 轴
            outputIdx += coord * tiling_->outputStrides[i];
        }
    }
    
    return outputIdx;
}

__aicore__ inline void KernelReduceAll::TileCopyIn(uint64_t gmOffset, uint32_t copySize) {
    LocalTensor<int8_t> inputLocal = inputQueue_.AllocTensor<int8_t>();

    DataCopyExtParams readParams;
    readParams.blockCount = 1;
    readParams.blockLen = copySize;
    readParams.srcStride = 0;
    readParams.dstStride = 0;

    DataCopyPadExtParams<int8_t> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad(inputLocal, inputGM_[gmOffset], readParams, padParams);
    inputQueue_.EnQue(inputLocal);
}


/*
__aicore__ inline void KernelReduceAll::CopyWorkGmToOutput() {
    // 重新分配 pipe
    pipe_.Destroy();
    pipe_.Init();
    pipe_.InitBuffer(outputQueue_, BUFFER_NUM, tileSize_ * sizeof(int8_t));   // 2

    uint64_t processedCount = 0;
    while (processedCount < totalOutputSize_) {
        const uint64_t remaining = totalOutputSize_ - processedCount;
        const uint32_t currentTileSize =
            (remaining < tileSize_) ? static_cast<uint32_t>(remaining) : tileSize_;

        LocalTensor<int8_t> outputLocal = outputQueue_.AllocTensor<int8_t>();

        const uint32_t int16Count = currentTileSize / 2;
        if (int16Count > 0) {
            LocalTensor<int16_t> outLocalInt16 = outputLocal.template ReinterpretCast<int16_t>();
            Duplicate(outLocalInt16, static_cast<int16_t>(0x0101), int16Count);
        }
        if (currentTileSize & 1U) {
            outputLocal.SetValue(currentTileSize - 1, static_cast<int8_t>(1));
        }

        for (uint32_t i = 0; i < currentTileSize; ++i) {     //  这段是否有必要 使用DataCopy 优化？ 
            const int8_t v = workGm_.GetValue(processedCount + static_cast<uint64_t>(i));
            outputLocal.SetValue(i, v);
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen   = currentTileSize;
        copyParams.srcStride  = 0;
        copyParams.dstStride  = 0;

        outputQueue_.EnQue(outputLocal);
        LocalTensor<int8_t> outLocal = outputQueue_.DeQue<int8_t>();

        SetAtomicMin<int8_t>();
        DataCopyPad(outputGM_[processedCount], outLocal, copyParams);
        SetAtomicNone();

        outputQueue_.FreeTensor(outLocal);

        processedCount += currentTileSize;
    }
}

*/

__aicore__ inline void KernelReduceAll::CopyWorkGmToOutput() {
    // ================================================================
    // 关键：将主计算阶段通过 SetValue 写入 workGm_ 的数据从 cache 刷到 GM
    // 否则后续 DataCopyPad 读 workGm_ 会读到初始化时的旧值（全1）
    // ================================================================
    DataCacheCleanAndInvalid<int8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workGm_);

    // 重建 pipe（释放主计算阶段的 buffer，分配搬运阶段的 buffer）
    pipe_.Destroy();
    pipe_.Init();

    // 分配两个 queue：一个用于从 workGm_ 读入，一个用于向 outputGM_ 写出
    // 这里复用 inputQueue_ 做读入，outputQueue_ 做写出
    pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileSize_ * sizeof(int8_t));
    pipe_.InitBuffer(outputQueue_, BUFFER_NUM, tileSize_ * sizeof(int8_t));

    uint64_t processedCount = 0;
    while (processedCount < totalOutputSize_) {
        const uint64_t remaining = totalOutputSize_ - processedCount;
        const uint32_t currentTileSize =
            (remaining < tileSize_) ? static_cast<uint32_t>(remaining) : tileSize_;

        // ========== Stage 1: DMA 从 workGm_ 搬入 UB ==========
        LocalTensor<int8_t> inLocal = inputQueue_.AllocTensor<int8_t>();

        DataCopyExtParams readParams;
        readParams.blockCount = 1;
        readParams.blockLen   = currentTileSize;
        readParams.srcStride  = 0;
        readParams.dstStride  = 0;

        DataCopyPadExtParams<int8_t> padParams;
        padParams.isPad        = true;
        padParams.leftPadding  = 0;
        padParams.rightPadding = static_cast<uint16_t>(
            (currentTileSize % 32 == 0) ? 0 : (32 - currentTileSize % 32));
        padParams.paddingValue = 1;  // padding 填 1，AtomicMin 对已有值无害

        DataCopyPad(inLocal, workGm_[processedCount], readParams, padParams);
        inputQueue_.EnQue(inLocal);

        // ========== Stage 2: UB 内搬运（inputQueue → outputQueue）==========
        LocalTensor<int8_t> inData = inputQueue_.DeQue<int8_t>();
        LocalTensor<int8_t> outLocal = outputQueue_.AllocTensor<int8_t>();

        // 用向量拷贝将数据从 inLocal 搬到 outLocal
        // 对齐到 32 字节的拷贝长度
        uint32_t copyLen32 = ((currentTileSize + 31) / 32) * 32;
        DataCopy(outLocal, inData, copyLen32);

        inputQueue_.FreeTensor(inData);
        outputQueue_.EnQue(outLocal);

        // ========== Stage 3: DMA 从 UB 搬出到 outputGM_（AtomicMin）==========
        LocalTensor<int8_t> outData = outputQueue_.DeQue<int8_t>();

        DataCopyExtParams writeParams;
        writeParams.blockCount = 1;
        writeParams.blockLen   = currentTileSize;
        writeParams.srcStride  = 0;
        writeParams.dstStride  = 0;

        SetAtomicMin<int8_t>();
        DataCopyPad(outputGM_[processedCount], outData, writeParams);
        SetAtomicNone();

        outputQueue_.FreeTensor(outData);

        processedCount += currentTileSize;
    }
}



__aicore__ inline void KernelReduceAll::ProcessAllReduce() {

    LocalTensor<half> workLocal = workBuf_.Get<half>();
    LocalTensor<half> reduceLocal = minBuf_.Get<half>();

    int8_t localMin = 1;
    uint64_t processedCount = 0;

    while (processedCount < myCount_) {
        uint32_t remaining = static_cast<uint32_t>(myCount_ - processedCount);
        uint32_t currentTileSize = (remaining < tileSize_) ? remaining : tileSize_;

        uint64_t gmOffset = myStart_ + processedCount;
        TileCopyIn(gmOffset, currentTileSize);

        LocalTensor<int8_t> inputLocal = inputQueue_.DeQue<int8_t>();
        
        Cast(workLocal, inputLocal, RoundMode::CAST_NONE, currentTileSize);
        inputQueue_.FreeTensor(inputLocal);
        Abs(workLocal, workLocal, currentTileSize);
        ReduceMin(reduceLocal, workLocal, workLocal, currentTileSize);

        half minVal = reduceLocal.GetValue(0);
        if ((float)minVal < (float)localMin) {
            localMin = 0;
            break;
        }
        
        processedCount += currentTileSize;
    }

   if (localMin == 0) {
        outputGM_.SetValue(0, static_cast<int8_t>(0));
    }
}

__aicore__ inline void KernelReduceAll::ProcessKR() {
    if (myCount_ == 0) {
        return;
    }

    LocalTensor<half> workLocal = workBuf_.Get<half>();
    LocalTensor<half> tmpLocal = tmpBuf_.Get<half>();
    LocalTensor<half> reduceLocal = minBuf_.Get<half>();

    // KR: [K][R]
    const uint32_t outerK = tiling_->fusedDims[0];
    const uint32_t innerR = tiling_->fusedDims[1];

    uint64_t processedCount = 0;

    while (processedCount < myCount_) {
        uint32_t remaining = static_cast<uint32_t>(myCount_ - processedCount);
        uint32_t currentTileSize = (remaining < tileSize_) ? remaining : tileSize_;

        const uint64_t gmOffset = myStart_ + processedCount;
        TileCopyIn(gmOffset, currentTileSize);

        LocalTensor<int8_t> inputLocal = inputQueue_.DeQue<int8_t>();

        Cast(workLocal, inputLocal, RoundMode::CAST_NONE, currentTileSize);
        Abs(workLocal, workLocal, currentTileSize);
        inputQueue_.FreeTensor(inputLocal);

        ReduceMin(reduceLocal, workLocal, tmpLocal, currentTileSize);
        half minVal = reduceLocal.GetValue(0);
        if ((float)minVal > 0.0f) {
            processedCount += currentTileSize;
            continue;
        }

        const uint64_t globalStart = myStart_ + processedCount;
        const uint64_t globalEnd   = globalStart + currentTileSize - 1;

        uint32_t startK = static_cast<uint32_t>(globalStart / innerR);
        uint32_t endK   = static_cast<uint32_t>(globalEnd   / innerR);
        if (endK >= outerK) {
            endK = outerK - 1;
        }


        for (uint32_t k = startK; k <= endK; ++k) {
            if (workGm_.GetValue(k) == 0) {      // 此处 GetValue 成本还是很高，可以考虑使用 tileBitMap 做过滤判断
                continue;
            }

            const uint64_t kGlobalStart = static_cast<uint64_t>(k) * innerR;
            const uint64_t kGlobalEnd   = kGlobalStart + innerR - 1;

            const uint64_t segStartG = (kGlobalStart > globalStart) ? kGlobalStart : globalStart;
            const uint64_t segEndG   = (kGlobalEnd   < globalEnd)   ? kGlobalEnd   : globalEnd;

            if (segStartG > segEndG) {
                continue;
            }

            const uint32_t segStartL = static_cast<uint32_t>(segStartG - globalStart);
            const uint32_t segLen    = static_cast<uint32_t>(segEndG - segStartG + 1);

            const uint32_t alignedStartL = AlignDown(segStartL, 32 / sizeof(half));
            const uint32_t prefix = segStartL - alignedStartL;

            if (prefix > 0) {
                Duplicate(workLocal[alignedStartL], static_cast<half>(1.0f), prefix);
            }

            const uint32_t reduceLen = prefix + segLen;

            ReduceMin(reduceLocal, workLocal[alignedStartL], workLocal[alignedStartL], reduceLen);

            half minVal = reduceLocal.GetValue(0);
            if (static_cast<float>(minVal) == 0.0f) {
                workGm_.SetValue(k, static_cast<int8_t>(0));
            }
        }

        processedCount += currentTileSize;
    }
}

__aicore__ inline void KernelReduceAll::ProcessRK() {
    if (myCount_ == 0) {
        return;
    }

    LocalTensor<half> workLocal = workBuf_.Get<half>();
    LocalTensor<half> tmpLocal = tmpBuf_.Get<half>();
    LocalTensor<half> reduceLocal = minBuf_.Get<half>();

    uint32_t outerR = tiling_->fusedDims[0];
    uint32_t innerK = tiling_->fusedDims[1];

    uint64_t processedCount = 0;

    while (processedCount < myCount_) {
        uint32_t remaining = static_cast<uint32_t>(myCount_ - processedCount);
        uint32_t currentTileSize = (remaining < tileSize_) ? remaining : tileSize_;
        
        uint64_t gmOffset = myStart_ + processedCount;
        TileCopyIn(gmOffset, currentTileSize);

        LocalTensor<int8_t> inputLocal = inputQueue_.DeQue<int8_t>();
        
        Cast(workLocal, inputLocal, RoundMode::CAST_NONE, currentTileSize);
        Abs(workLocal, workLocal, currentTileSize);
        inputQueue_.FreeTensor(inputLocal);

        ReduceMin(reduceLocal, workLocal, tmpLocal, currentTileSize);
        half minVal = reduceLocal.GetValue(0);
        if ((float)minVal > 0.0f) {
            processedCount += currentTileSize;
            continue;
        }

        uint64_t globalStart = myStart_ + processedCount;
        uint64_t globalEnd = globalStart + currentTileSize - 1;
        
        // R 和 K 范围
        uint32_t startR = static_cast<uint32_t>(globalStart / innerK);
        uint32_t endR = static_cast<uint32_t>(globalEnd / innerK);
        uint32_t startK = static_cast<uint32_t>(globalStart % innerK);
        uint32_t endK = static_cast<uint32_t>(globalEnd % innerK);
        
        uint32_t kRangeStart, kRangeEnd;
        if (startR == endR) {
            kRangeStart = startK;
            kRangeEnd = endK;
        } else {
            kRangeStart = 0;
            kRangeEnd = innerK - 1;
        }

        // 遍历每个 K 位置
        for (uint32_t k = kRangeStart; k <= kRangeEnd; ++k) {
            if (workGm_.GetValue(k) == 0) {
                continue;
            }

            // 计算 k 在当前 tile 中第一次出现的位置
            uint32_t firstK = static_cast<uint32_t>(globalStart % innerK);
            uint32_t firstLocalIdx;
            if (firstK <= k) {
                firstLocalIdx = k - firstK;
            } else {
                firstLocalIdx = innerK - firstK + k;
            }
            
            // 以 innerK 为步长遍历
            for (uint32_t localIdx = firstLocalIdx; localIdx < currentTileSize; localIdx += innerK) {
                half val = workLocal.GetValue(localIdx);
                if (static_cast<float>(val) == 0.0f) {
                    workGm_.SetValue(k, static_cast<int8_t>(0));
                    break;
                }
            }
        }
        processedCount += currentTileSize;
    }

}

__aicore__ inline void KernelReduceAll::ProcessRKR() {
    if (myCount_ == 0) {
        return;
    }

    LocalTensor<half> workLocal = workBuf_.Get<half>();
    LocalTensor<half> tmpLocal = tmpBuf_.Get<half>();
    LocalTensor<half> reduceLocal = minBuf_.Get<half>();

    const uint32_t outerR1  = tiling_->fusedDims[0];
    const uint32_t middleK  = tiling_->fusedDims[1];
    const uint32_t innerR2  = tiling_->fusedDims[2];

    const uint64_t kR2Stride = static_cast<uint64_t>(middleK) * static_cast<uint64_t>(innerR2);

    if (middleK == 0 || innerR2 == 0 || kR2Stride == 0) {
        return;
    }

    uint64_t processedCount = 0;

    while (processedCount < myCount_) {
        uint32_t remaining = static_cast<uint32_t>(myCount_ - processedCount);
        uint32_t currentTileSize = (remaining < tileSize_) ? remaining : tileSize_;

        const uint64_t gmOffset = myStart_ + processedCount;
        TileCopyIn(gmOffset, currentTileSize);

        LocalTensor<int8_t> inputLocal = inputQueue_.DeQue<int8_t>();

        Cast(workLocal, inputLocal, RoundMode::CAST_NONE, currentTileSize);
        Abs(workLocal, workLocal, currentTileSize);
        inputQueue_.FreeTensor(inputLocal);

        ReduceMin(reduceLocal, workLocal, tmpLocal, currentTileSize);
        half minVal = reduceLocal.GetValue(0);
        if ((float)minVal > 0.0f) {
            processedCount += currentTileSize;
            continue;
        }

        const uint64_t globalStart = myStart_ + processedCount;

        uint32_t lastK = 0xFFFFFFFFu;
        bool lastOutAlreadyZero = false;

        for (uint32_t localIdx = 0; localIdx < currentTileSize; ++localIdx) {
            half v = workLocal.GetValue(localIdx);
            if (static_cast<float>(v) != 0.0f) {
                continue;
            }

            const uint64_t globalIdx = globalStart + static_cast<uint64_t>(localIdx);

            // globalIdx = r1*(K*R2) + k*R2 + r2
            // k = (globalIdx % (K*R2)) / R2
            const uint64_t r1 = globalIdx / kR2Stride;
            if (r1 >= static_cast<uint64_t>(outerR1)) {
                continue;
            }

            const uint64_t inR1Offset = globalIdx - r1 * kR2Stride;
            const uint32_t k = static_cast<uint32_t>(inR1Offset / static_cast<uint64_t>(innerR2));

            if (k == lastK && lastOutAlreadyZero) {
                continue;
            }

            if (workGm_.GetValue(k) == 0) {
                lastK = k;
                lastOutAlreadyZero = true;
                continue;
            }

            workGm_.SetValue(k, static_cast<int8_t>(0));

            lastK = k;
            lastOutAlreadyZero = true;
        }

        processedCount += currentTileSize;
    }
}

__aicore__ inline void KernelReduceAll::ProcessKRK() {
    if (myCount_ == 0) {
        return;
    }

    LocalTensor<half> workLocal = workBuf_.Get<half>();
    LocalTensor<half> tmpLocal = tmpBuf_.Get<half>();
    LocalTensor<half> reduceLocal = minBuf_.Get<half>();

    const uint32_t outerK1  = tiling_->fusedDims[0];
    const uint32_t middleR  = tiling_->fusedDims[1];
    const uint32_t innerK2  = tiling_->fusedDims[2];

    const uint64_t rK2Stride = static_cast<uint64_t>(middleR) * static_cast<uint64_t>(innerK2);

    uint64_t processedCount = 0;

    while (processedCount < myCount_) {
        uint32_t remaining = static_cast<uint32_t>(myCount_ - processedCount);
        uint32_t currentTileSize = (remaining < tileSize_) ? remaining : tileSize_;

        const uint64_t gmOffset = myStart_ + processedCount;
        TileCopyIn(gmOffset, currentTileSize);

        LocalTensor<int8_t> inputLocal = inputQueue_.DeQue<int8_t>();

        Cast(workLocal, inputLocal, RoundMode::CAST_NONE, currentTileSize);
        Abs(workLocal, workLocal, currentTileSize);
        inputQueue_.FreeTensor(inputLocal);

        ReduceMin(reduceLocal, workLocal, tmpLocal, currentTileSize);
        half minVal = reduceLocal.GetValue(0);
        if ((float)minVal > 0.0f) {
            processedCount += currentTileSize;
            continue;
        }

        const uint64_t globalStart = myStart_ + processedCount;

        uint32_t lastK1 = 0xFFFFFFFFu;
        uint32_t lastK2 = 0xFFFFFFFFu;
        bool lastOutAlreadyZero = false;

        for (uint32_t localIdx = 0; localIdx < currentTileSize; ++localIdx) {
            half v = workLocal.GetValue(localIdx);
            if (static_cast<float>(v) != 0.0f) {
                continue;
            }

            // 这里能否 使用很低的成本找到下一个0的位置？

            const uint64_t globalIdx = globalStart + static_cast<uint64_t>(localIdx);
            const uint32_t k1 = static_cast<uint32_t>(globalIdx / rK2Stride);
            
            if (k1 >= outerK1) {
                continue;
            }

            const uint32_t k2 = static_cast<uint32_t>(globalIdx % innerK2);
            const uint32_t outputIdx = k1 * innerK2 + k2;

            if (k1 == lastK1 && k2 == lastK2 && lastOutAlreadyZero) {
                continue;
            }

            if (workGm_.GetValue(outputIdx) == 0) {
                lastK1 = k1;
                lastK2 = k2;
                lastOutAlreadyZero = true;
                continue;
            }
            
            workGm_.SetValue(outputIdx, static_cast<int8_t>(0));

            lastK1 = k1;
            lastK2 = k2;
            lastOutAlreadyZero = true;

        }

        processedCount += currentTileSize;
    }
}

__aicore__ inline void KernelReduceAll::ProcessGeneral() {

    LocalTensor<half> workLocal = workBuf_.Get<half>();
    LocalTensor<half> tmpLocal = tmpBuf_.Get<half>();
    LocalTensor<half> reduceLocal = minBuf_.Get<half>();

    const uint32_t fusedDimCount = tiling_->fusedDimCount;
    const uint32_t lastDim = (fusedDimCount == 0) ? 0 : (fusedDimCount - 1);

    const bool lastIsR = (fusedDimCount > 0) && (tiling_->fusedAxes[lastDim] != 0);
    const uint32_t innerR = lastIsR ? tiling_->fusedDims[lastDim] : 1;

    uint64_t processedCount = 0;

    while (processedCount < myCount_) {
        uint32_t remaining = static_cast<uint32_t>(myCount_ - processedCount);
        uint32_t currentTileSize = (remaining < tileSize_) ? remaining : tileSize_;

        uint64_t gmOffset = myStart_ + processedCount;
        TileCopyIn(gmOffset, currentTileSize);

        LocalTensor<int8_t> inputLocal = inputQueue_.DeQue<int8_t>();
        
        Cast(workLocal, inputLocal, RoundMode::CAST_NONE, currentTileSize);
        Abs(workLocal, workLocal, currentTileSize);
        inputQueue_.FreeTensor(inputLocal);

        ReduceMin(reduceLocal, workLocal, tmpLocal, currentTileSize);
        half minVal = reduceLocal.GetValue(0);
        if ((float)minVal > 0.0f) {
            processedCount += currentTileSize;
            continue;
        }
        
        const uint64_t globalStart = myStart_ + processedCount;
        for (uint32_t localIdx = 0; localIdx < currentTileSize; ++localIdx) {
            half v = workLocal.GetValue(localIdx);
            if (static_cast<float>(v) != 0.0f) continue;

            const uint64_t globalIdx = globalStart + static_cast<uint64_t>(localIdx);
            const uint32_t outputIdx = GlobalIdxToOutputIdx(globalIdx);
            workGm_.SetValue(outputIdx, static_cast<int8_t>(0));

            if (lastIsR && innerR > 1) {
                uint32_t offsetInR = static_cast<uint32_t>(globalIdx % innerR);
                uint32_t remainInR = innerR - 1 - offsetInR;
                if (remainInR > 0) {
                    uint32_t maxRemainInTile = currentTileSize - 1 - localIdx;
                    uint32_t jump = (remainInR < maxRemainInTile) ? remainInR : maxRemainInTile;
                    localIdx += jump;
                }
            }
        }

        processedCount += currentTileSize;
    }
    workBuf_.FreeTensor(workLocal);
}

}  // namespace NsReduceAll

#endif  // REDUCE_ALL_H_
