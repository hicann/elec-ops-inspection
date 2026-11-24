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

#ifndef FDK_ONCHIP_JOIN_CUBE_H
#define FDK_ONCHIP_JOIN_CUBE_H

#include "fdk_common.h"

namespace fdk {
// On-chip horizontal recombination on Cube using compensated two-half FP16 split.
template <int32_t W, int32_t ROWS>
class FdkOnchipJoinCube {
    static constexpr int32_t N = W * ROWS;
    static constexpr int32_t K = 96;
    static constexpr int32_t T = 18 * W;

public:
    __aicore__ inline void Run(
        GM_ADDR feat,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        InitBuffers(feat, scratch, t);
        LoadHaloAndWeights(t);
        for (int32_t y = begin_; y < end_; y += ROWS) {
            ComputeProjectionTile(y);
            EmitCompensatedJoin(y);
        }
        AscendC::CrossCoreWaitFlag<2>(9);
    }

private:
    __aicore__ inline void InitBuffers(
        GM_ADDR feat,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        pipe_.InitBuffer(features_, T * 32 * 4);
        pipe_.InitBuffer(weights_, 16 * K * 4);
        pipe_.InitBuffer(halves_, 2 * N * 16 * 2);
        pipe_.InitBuffer(constants_, 1024 * 2);
        pipe_.InitBuffer(aa_, N * 48 * 4);
        pipe_.InitBuffer(bb_, 8192);
        pipe_.InitBuffer(cc_, 2 * 16 * N * 4);
        f_ = features_.Get<float>();
        w_ = weights_.Get<float>();
        a_ = aa_.Get<float>();
        b_ = bb_.Get<float>();
        c_ = cc_.Get<float>();
        hi_ = halves_.Get<half>();
        lo_ = hi_[N * 16];
        id_ = constants_.Get<half>();
        join_ = id_[256];
        ah_ = a_.ReinterpretCast<half>();
        idb_ = b_.ReinterpretCast<half>()[3072];
        jb_ = idb_[256];
        result_ = c_[16 * N];
        fg_.SetGlobalBuffer((__gm__ float *)feat);
        packed_.SetGlobalBuffer((__gm__ float *)scratch);
        ph_.SetGlobalBuffer((__gm__ half *)scratch);
        const int32_t group = static_cast<int32_t>(AscendC::GetBlockIdx());
        const int32_t groupsPerBatch = static_cast<int32_t>(t.groupsPerBatch);
        const int32_t h = static_cast<int32_t>(t.height);
        batch_ = group / groupsPerBatch;
        ring_.SetGlobalBuffer((__gm__ float *)scratch + 1408 + group * static_cast<int32_t>(t.groupFloats));
        const int32_t part = ((h / ROWS) + groupsPerBatch - 1) / groupsPerBatch;
        begin_ = (group % groupsPerBatch) * part * ROWS;
        const int32_t candEnd = begin_ + part * ROWS;
        end_ = (candEnd < h) ? candEnd : h;
    }

    __aicore__ inline void LoadHaloAndWeights(const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        const int32_t h = static_cast<int32_t>(t.height);
        const int32_t s = h * W;
        const int32_t proposed = begin_ - 1;
        const int32_t start = (proposed < 0) ? 0 : proposed;
        const int32_t stop = (end_ + 1 < h) ? (end_ + 1) : h;
        const int32_t valid = (stop - start) * W;
        const int32_t top = (start - proposed) * W;
        if (valid < T) {
            AscendC::Fill(f_, AscendC::InitConstValueParams<float>{1, static_cast<uint32_t>(T * 32 / 8), 0, 0.0f});
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::Nd2NzParams ft{};
        ft.ndNum = 1;
        ft.nValue = 16;
        ft.dValue = static_cast<uint16_t>(valid);
        ft.srcDValue = static_cast<uint16_t>(s);
        ft.dstNzC0Stride = 16;
        ft.dstNzNStride = 1;
        AscendC::DataCopy(f_[top * 16], fg_[batch_ * 32 * s + start * W], ft);
        AscendC::DataCopy(f_[16 * T + top * 16], fg_[(batch_ * 32 + 16) * s + start * W], ft);
        AscendC::SyncAll<false>();
        LoadPackedWeights();
    }

    __aicore__ inline void LoadPackedWeights()
    {
        AscendC::Nd2NzParams wt{};
        wt.ndNum = 1;
        wt.nValue = 12;
        wt.dValue = 96;
        wt.srcDValue = 96;
        wt.dstNzC0Stride = 16;
        wt.dstNzNStride = 1;
        AscendC::DataCopy(w_, packed_, wt);
        AscendC::DataCopy(id_, ph_[2304], 256);
        AscendC::Nd2NzParams jt{};
        jt.ndNum = 1;
        jt.nValue = 4;
        jt.dValue = 48;
        jt.srcDValue = 48;
        jt.dstNzC0Stride = 16;
        jt.dstNzNStride = 1;
        AscendC::DataCopy(join_, ph_[2560], jt);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID6);
        AscendC::LoadData2DParams wd{};
        wd.repeatTimes = 12;
        wd.srcStride = 1;
        AscendC::LoadData(b_, w_, wd);
        wd.repeatTimes = 1;
        AscendC::LoadData(idb_, id_, wd);
        wd.repeatTimes = 3;
        AscendC::LoadData(jb_, join_, wd);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
        AscendC::SetHF32Mode(false);
        AscendC::SetHF32TransMode(false);
    }

    __aicore__ inline void ComputeProjectionTile(int32_t y)
    {
        for (int32_t phase = 0; phase < 2; ++phase) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::LoadData2dTransposeParams tr{};
            tr.repeatTimes = static_cast<uint8_t>(N / 16);
            tr.srcStride = 1;
            tr.dstGap = 5;
            tr.dstFracGap = 0;
            for (int32_t j = 0; j < 3; ++j) {
                const int32_t comp = phase * 3 + j;
                const int32_t srcOff = (comp % 2) * 16 * T + (y - begin_ + comp / 2) * W * 16;
                AscendC::LoadDataWithTranspose(a_[j * 256], f_[srcOff], tr);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
            AscendC::MmadParams mm{};
            mm.m = static_cast<uint16_t>(N);
            mm.n = 16;
            mm.k = 48;
            mm.cmatrixInitVal = (phase == 0);
            AscendC::Mmad(c_, a_, b_[phase * 768], mm);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID5);
    }

    __aicore__ inline void EmitCompensatedJoin(int32_t y)
    {
        AscendC::FixpipeParamsV220 fp{};
        fp.nSize = 16;
        fp.mSize = static_cast<uint16_t>(N);
        fp.srcStride = static_cast<uint16_t>(N);
        fp.dstStride = static_cast<uint32_t>(N);
        fp.quantPre = AscendC::QuantMode_t::F322F16;
        AscendC::Fixpipe<half, float, AscendC::CFG_NZ>(hi_, c_, fp);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);
        AscendC::LoadData3DParamsV2<half> lh{};
        lh.l1H = static_cast<uint16_t>(ROWS);
        lh.l1W = static_cast<uint16_t>(W);
        lh.channelSize = 16;
        lh.mExtension = static_cast<uint16_t>(N);
        lh.kExtension = 16;
        AscendC::LoadData(ah_, hi_, lh);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
        AscendC::MmadParams sub{};
        sub.m = static_cast<uint16_t>(N);
        sub.n = 16;
        sub.k = 16;
        sub.cmatrixInitVal = false;
        AscendC::Mmad(c_, ah_, idb_, sub);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID5);
        AscendC::Fixpipe<half, float, AscendC::CFG_NZ>(lo_, c_, fp);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);
        RunHorizontalJoinAndOut(y, lh);
    }

    __aicore__ inline void RunHorizontalJoinAndOut(int32_t y, AscendC::LoadData3DParamsV2<half> &lh)
    {
        lh.padList[0] = 1;
        lh.padList[1] = 1;
        lh.filterW = 3;
        lh.kExtension = 48;
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::LoadData(ah_, hi_, lh);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
        if (y != begin_) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
        }
        AscendC::MmadParams merge{};
        merge.m = static_cast<uint16_t>(N);
        merge.n = 4;
        merge.k = 48;
        merge.cmatrixInitVal = true;
        AscendC::Mmad(result_, ah_, jb_, merge);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::LoadData(ah_, lo_, lh);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID7);
        merge.cmatrixInitVal = false;
        AscendC::Mmad(result_, ah_, jb_, merge);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID5);
        if (y != begin_) {
            AscendC::CrossCoreWaitFlag<2>(9);
        }
        AscendC::FixpipeParamsV220 out{};
        out.nSize = 4;
        out.mSize = static_cast<uint16_t>(N);
        out.srcStride = static_cast<uint16_t>(N);
        out.dstStride = 4;
        AscendC::Fixpipe(ring_, result_, out);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(2);
        if (y + ROWS < end_) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
        }
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> features_;
    AscendC::TBuf<AscendC::TPosition::A1> weights_;
    AscendC::TBuf<AscendC::TPosition::A1> halves_;
    AscendC::TBuf<AscendC::TPosition::A1> constants_;
    AscendC::TBuf<AscendC::TPosition::A2> aa_;
    AscendC::TBuf<AscendC::TPosition::B2> bb_;
    AscendC::TBuf<AscendC::TPosition::CO1> cc_;
    AscendC::LocalTensor<float> f_;
    AscendC::LocalTensor<float> w_;
    AscendC::LocalTensor<float> a_;
    AscendC::LocalTensor<float> b_;
    AscendC::LocalTensor<float> c_;
    AscendC::LocalTensor<float> result_;
    AscendC::LocalTensor<half> hi_;
    AscendC::LocalTensor<half> lo_;
    AscendC::LocalTensor<half> id_;
    AscendC::LocalTensor<half> join_;
    AscendC::LocalTensor<half> ah_;
    AscendC::LocalTensor<half> idb_;
    AscendC::LocalTensor<half> jb_;
    AscendC::GlobalTensor<float> fg_;
    AscendC::GlobalTensor<float> packed_;
    AscendC::GlobalTensor<float> ring_;
    AscendC::GlobalTensor<half> ph_;
    int32_t batch_ = 0;
    int32_t begin_ = 0;
    int32_t end_ = 0;
};
}  // namespace fdk

#endif  // FDK_ONCHIP_JOIN_CUBE_H
