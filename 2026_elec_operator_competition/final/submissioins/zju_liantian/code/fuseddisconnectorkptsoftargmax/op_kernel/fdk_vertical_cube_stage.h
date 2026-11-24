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

#ifndef FDK_VERTICAL_CUBE_STAGE_H
#define FDK_VERTICAL_CUBE_STAGE_H

#include "fdk_common.h"

namespace fdk {
// Locally adapted Cube schedule for B=1 48x64 and B=1 64x64 (authorized partial reuse).
// Host planning, weight packing, AIV reconstruction, statistics and merge are original.
template <int32_t W, uint32_t FLAG_WT, uint32_t FLAG_OUT, uint32_t FLAG_ACK>
class FdkVerticalCubeStage {
    static constexpr uint32_t P = static_cast<uint32_t>(2 * W);

public:
    __aicore__ inline void Run(
        GM_ADDR feat,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        InitAndLoad(feat, scratch, t);
        for (uint32_t r = 0U; r < rows_; r += 2U) {
            ProcessRowPair(r);
        }
        FinishSync();
    }

private:
    __aicore__ inline void InitAndLoad(
        GM_ADDR feat,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        const uint32_t h = t.height;
        const uint32_t w = static_cast<uint32_t>(W);
        const uint32_t full = h * w;
        const uint32_t parts = t.groupsPerBatch;
        const uint32_t group = AscendC::GetBlockIdx();
        const uint32_t part = group % parts;
        const uint32_t b = group / parts;
        const uint32_t first = (part * (h / 2U) / parts) * 2U;
        rows_ = (((part + 1U) * (h / 2U) / parts) * 2U) - first;
        halo_ = (rows_ + 2U) * w;
        pipe_.InitBuffer(inputBuf_, 32U * halo_ * 4U);
        pipe_.InitBuffer(weightBuf_, 16U * 96U * 4U);
        pipe_.InitBuffer(aBuf_, 16U * 96U * 4U);
        pipe_.InitBuffer(bBuf_, 96U * P * 4U);
        pipe_.InitBuffer(cBuf_, 2U * 16U * P * 4U);
        x_ = inputBuf_.Get<float>();
        wt_ = weightBuf_.Get<float>();
        a_ = aBuf_.Get<float>();
        xb_ = bBuf_.Get<float>();
        c_ = cBuf_.Get<float>();
        fg_.SetGlobalBuffer((__gm__ float *)feat);
        ring_.SetGlobalBuffer((__gm__ float *)scratch + group * t.groupFloats);
        AscendC::SetHF32Mode(false);
        AscendC::SetHF32TransMode(false);
        const uint32_t start = (first != 0U) ? (first - 1U) : 0U;
        const uint32_t top = (first != 0U) ? 0U : w;
        const uint32_t end = (first + rows_ < h) ? (first + rows_ + 1U) : h;
        const uint32_t valid = (end - start) * w;
        LoadHaloFeatures(b, full, start, top, valid, first + rows_ == h);
        LoadWeightsToL0A();
    }

    __aicore__ inline void LoadHaloFeatures(
        uint32_t b,
        uint32_t full,
        uint32_t start,
        uint32_t top,
        uint32_t valid,
        bool isBottomEdge)
    {
        const uint32_t w = static_cast<uint32_t>(W);
        const uint16_t padBlocks = static_cast<uint16_t>(32U * w / 8U);
        if (top != 0U) {
            AscendC::InitConstValue(x_, AscendC::InitConstValueParams<float>{1, padBlocks, 0, 0.0f});
        }
        if (isBottomEdge) {
            AscendC::InitConstValue(
                x_[(halo_ - w) * 32U],
                AscendC::InitConstValueParams<float>{1, padBlocks, 0, 0.0f});
        }
        if ((top != 0U) || isBottomEdge) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::Nd2NzParams nd{};
        nd.ndNum = 1;
        nd.nValue = 32;
        nd.dValue = static_cast<uint16_t>(valid);
        nd.srcDValue = static_cast<uint16_t>(full);
        nd.dstNzC0Stride = 32;
        nd.dstNzNStride = 1;
        AscendC::DataCopy(x_[top * 32U], fg_[b * 32U * full + start * w], nd);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadWeightsToL0A()
    {
        AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(FLAG_WT);
        AscendC::Nd2NzParams nw{};
        nw.ndNum = 1;
        nw.nValue = 12;
        nw.dValue = 96;
        nw.srcDValue = 96;
        nw.dstNzC0Stride = 16;
        nw.dstNzNStride = 1;
        AscendC::DataCopy(wt_, ring_, nw);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LoadData2DParams aw{};
        aw.repeatTimes = 12;
        aw.srcStride = 1;
        AscendC::LoadData(a_, wt_, aw);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadRowPairToL0B(uint32_t r)
    {
        for (uint32_t dy = 0U; dy < 3U; ++dy) {
            AscendC::LoadData3DParamsV2<float> pb{};
            pb.l1H = 1;
            pb.l1W = 32;
            pb.channelSize = static_cast<uint16_t>(halo_);
            pb.kExtension = static_cast<uint16_t>(P);
            pb.mExtension = 32;
            pb.kStartPt = static_cast<uint16_t>((r + dy) * static_cast<uint32_t>(W));
            pb.enTranspose = true;
            pb.filterW = 1;
            pb.filterH = 1;
            pb.strideW = 1;
            pb.strideH = 1;
            pb.dilationFilterW = 1;
            pb.dilationFilterH = 1;
            AscendC::LoadData(xb_[dy * 32U * P], x_, pb);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID5);
    }

    __aicore__ inline void ProcessRowPair(uint32_t r)
    {
        if (r != 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        }
        LoadRowPairToL0B(r);
        const uint32_t slot = (r / 2U) % 2U;
        if (r >= 4U) {
            if (slot == 0U) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID6);
            }
        }
        auto tileC = c_[slot * 16U * P];
        AscendC::MmadParams mm{};
        mm.m = 16;
        mm.n = static_cast<uint16_t>(P);
        mm.k = 96;
        mm.cmatrixInitVal = true;
        AscendC::Mmad(tileC, a_, xb_, mm);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID5);
        if (r != 0U) {
            AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_ACK);
        }
        AscendC::FixpipeParamsV220 fp{};
        fp.mSize = 12;
        fp.nSize = static_cast<uint16_t>(P);
        fp.srcStride = 16;
        fp.dstStride = P;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(ring_, tileC, fp);
        if (slot == 0U) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID6);
        }
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_OUT);
    }

    __aicore__ inline void FinishSync()
    {
        if (rows_ > 0U) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
        }
        if (rows_ > 2U) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID6);
        }
        AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_ACK);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::B1> weightBuf_;
    AscendC::TBuf<AscendC::TPosition::A2> aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> bBuf_;
    AscendC::TBuf<AscendC::TPosition::CO1> cBuf_;
    AscendC::LocalTensor<float> x_;
    AscendC::LocalTensor<float> wt_;
    AscendC::LocalTensor<float> a_;
    AscendC::LocalTensor<float> xb_;
    AscendC::LocalTensor<float> c_;
    AscendC::GlobalTensor<float> fg_;
    AscendC::GlobalTensor<float> ring_;
    uint32_t rows_ = 0U;
    uint32_t halo_ = 0U;
};
}  // namespace fdk

#endif  // FDK_VERTICAL_CUBE_STAGE_H
