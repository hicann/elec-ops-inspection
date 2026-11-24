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

#ifndef FDK_PROJECTION_VECTOR_BASE_H
#define FDK_PROJECTION_VECTOR_BASE_H

#include "fdk_common.h"

namespace fdk {
class FdkProjectionVectorBase {
protected:
    __aicore__ inline void InitProjectionState(
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &td,
        int32_t tileRows,
        int32_t tilePixels,
        int32_t maxAcc)
    {
        t_ = td;
        group_ = static_cast<int32_t>(AscendC::GetBlockIdx()) / 2;
        sub_ = static_cast<int32_t>(AscendC::GetSubBlockIdx());
        const int32_t rawGpb = static_cast<int32_t>(t_.groupsPerBatch);
        const int32_t gpb = (rawGpb > 0) ? rawGpb : 1;
        const int32_t stepRows = (tileRows > 0) ? tileRows : 1;
        const int32_t h = static_cast<int32_t>(t_.height);
        const int32_t tiles = (h + stepRows - 1) / stepRows;
        const int32_t per = (tiles + gpb - 1) / gpb;
        first_ = (group_ % gpb) * per * stepRows;
        const int32_t candLast = first_ + per * stepRows;
        last_ = (candLast < h) ? candLast : h;
        rows_ = last_ - first_;
        wg_.SetGlobalBuffer((__gm__ float *)weight);
        bg_.SetGlobalBuffer((__gm__ float *)bias);
        const int32_t gf = static_cast<int32_t>(t_.groupFloats);
        ring_.SetGlobalBuffer((__gm__ float *)scratch + group_ * gf);
        stats_.SetGlobalBuffer((__gm__ float *)scratch + static_cast<int32_t>(t_.batch) * gpb * gf);
        InitProjectionBuffers(tilePixels, maxAcc);
    }

    __aicore__ inline void InitProjectionBuffers(int32_t tilePixels, int32_t maxAcc)
    {
        const int32_t rawElems = (2 * (6 * tilePixels + 16) > maxAcc) ? (2 * (6 * tilePixels + 16)) : maxAcc;
        const int32_t tmpElems = (6 * tilePixels > maxAcc) ? (6 * tilePixels) : maxAcc;
        const int32_t idxElems = (6 * tilePixels > 576) ? (6 * tilePixels) : 576;
        pipe_.InitBuffer(rawBuf_, rawElems * 4);
        pipe_.InitBuffer(accBuf_, 2 * maxAcc * 4);
        pipe_.InitBuffer(tmpBuf_, tmpElems * 4);
        pipe_.InitBuffer(indexBuf_, idxElems * 4);
        pipe_.InitBuffer(smallBuf_, 128 * 4);
        pipe_.InitBuffer(coordBuf_, maxAcc * 4);
        raw_ = rawBuf_.Get<float>()[8];
        acc_ = accBuf_.Get<float>();
        temp_ = tmpBuf_.Get<float>();
        ix_ = indexBuf_.Get<uint32_t>();
        small_ = smallBuf_.Get<float>();
        coord_ = coordBuf_.Get<float>();
    }

    __aicore__ inline void AccumulateChannelPair(
        int32_t rowOffset,
        int32_t count,
        int32_t width,
        int32_t tilePixels,
        int32_t maxAcc)
    {
        for (int32_t k = 0; k < 2; ++k) {
            auto dst = acc_[k * maxAcc + rowOffset * width];
            AscendC::Add(dst, temp_[k * 3 * tilePixels], temp_[(k * 3 + 1) * tilePixels], count * width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(dst, dst, temp_[(k * 3 + 2) * tilePixels], count * width);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreChannelStats(int32_t k, float peak, float den, float sx, float sy)
    {
        small_.SetValue(32, peak);
        small_.SetValue(33, den);
        small_.SetValue(34, sx);
        small_.SetValue(35, sy);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopy(stats_[group_ * 32 + (sub_ * 2 + k) * 8], small_[32], 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> smallBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> coordBuf_;
    AscendC::LocalTensor<float> raw_;
    AscendC::LocalTensor<float> acc_;
    AscendC::LocalTensor<float> temp_;
    AscendC::LocalTensor<float> small_;
    AscendC::LocalTensor<float> coord_;
    AscendC::LocalTensor<uint32_t> ix_;
    AscendC::GlobalTensor<float> wg_;
    AscendC::GlobalTensor<float> bg_;
    AscendC::GlobalTensor<float> ring_;
    AscendC::GlobalTensor<float> stats_;
    FusedDisconnectorKeypointSoftArgmaxTilingData t_{};
    int32_t group_ = 0;
    int32_t sub_ = 0;
    int32_t first_ = 0;
    int32_t last_ = 0;
    int32_t rows_ = 0;
};
}  // namespace fdk

#endif  // FDK_PROJECTION_VECTOR_BASE_H
