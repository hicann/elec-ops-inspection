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

#ifndef FDK_DIRECT_PROJECTION_H
#define FDK_DIRECT_PROJECTION_H

#include "fdk_projection_vector_base.h"

namespace fdk {
template <int32_t W, int32_t R, int32_t PARTROWS>
class DirectProjectionCube {
    static constexpr int32_t P = R * W;
    static constexpr int32_t HALO = (PARTROWS + 2) * W;

public:
    __aicore__ inline void Run(
        GM_ADDR feat,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        InitBuffers(feat, scratch, t);
        LoadHaloAndWeights(t);
        for (int32_t y = first_; y < last_; y += R) {
            ProcessRowStrip(y);
        }
        FinishSync();
    }

private:
    __aicore__ inline void InitBuffers(
        GM_ADDR feat,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        pipe_.InitBuffer(al_, 16 * 96 * 4);
        pipe_.InitBuffer(bl_, 32 * HALO * 4);
        pipe_.InitBuffer(a0b_, 16 * 96 * 4);
        pipe_.InitBuffer(b0b_, 96 * P * 4);
        pipe_.InitBuffer(cb_, 2 * 16 * P * 4);
        a1_ = al_.Get<float>();
        b1_ = bl_.Get<float>();
        a0_ = a0b_.Get<float>();
        b0_ = b0b_.Get<float>();
        c_ = cb_.Get<float>();
        const int32_t g = static_cast<int32_t>(AscendC::GetBlockIdx());
        const int32_t gpb = static_cast<int32_t>(t.groupsPerBatch);
        const int32_t h = static_cast<int32_t>(t.height);
        batch_ = g / gpb;
        fg_.SetGlobalBuffer((__gm__ float *)feat);
        ring_.SetGlobalBuffer((__gm__ float *)scratch + g * static_cast<int32_t>(t.groupFloats));
        const int32_t tiles = (h + R - 1) / R;
        const int32_t per = (tiles + gpb - 1) / gpb;
        first_ = (g % gpb) * per * R;
        const int32_t candLast = first_ + per * R;
        last_ = (candLast < h) ? candLast : h;
    }

    __aicore__ inline void LoadHaloAndWeights(const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        const int32_t h = static_cast<int32_t>(t.height);
        const int32_t s = h * W;
        const int32_t proposed = first_ - 1;
        const int32_t start = (proposed < 0) ? 0 : proposed;
        const int32_t end = (last_ + 1 < h) ? (last_ + 1) : h;
        const int32_t valid = (end - start) * W;
        const int32_t top = (start - proposed) * W;
        if (valid < HALO) {
            AscendC::Fill(b1_, AscendC::InitConstValueParams<float>{1, static_cast<uint16_t>(32 * HALO / 8), 0, 0.0f});
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::Nd2NzParams p{};
        p.ndNum = 1;
        p.nValue = 32;
        p.dValue = static_cast<uint16_t>(valid);
        p.srcDValue = static_cast<uint16_t>(s);
        p.dstNzC0Stride = 32;
        p.dstNzNStride = 1;
        AscendC::DataCopy(b1_[top * 32], fg_[batch_ * 32 * s + start * W], p);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::CrossCoreWaitFlag<2>(1);
        AscendC::Fill(a1_, AscendC::InitConstValueParams<float>{1, 192, 0, 0.0f});
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::Nd2NzParams wa{};
        wa.ndNum = 1;
        wa.nValue = 12;
        wa.dValue = 96;
        wa.srcDValue = 96;
        wa.dstNzC0Stride = 16;
        wa.dstNzNStride = 1;
        AscendC::DataCopy(a1_, ring_, wa);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::LoadData2DParams q{};
        q.repeatTimes = 12;
        q.srcStride = 1;
        AscendC::LoadData(a0_, a1_, q);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetHF32Mode(false);
        AscendC::SetHF32TransMode(false);
    }

    __aicore__ inline void LoadStripToL0B(int32_t y, int32_t n)
    {
        AscendC::LoadData3DParamsV2<float> load{};
        load.l1H = 1;
        load.l1W = 32;
        load.channelSize = HALO;
        load.mExtension = 32;
        load.kExtension = static_cast<uint16_t>(n);
        load.enTranspose = false;
        for (int32_t dy = 0; dy < 3; ++dy) {
            load.kStartPt = static_cast<uint16_t>((y - first_ + dy) * W);
            AscendC::LoadData(b0_[dy * 32 * n], b1_, load);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    }

    __aicore__ inline void ProcessRowStrip(int32_t y)
    {
        const int32_t rows = (last_ - y < R) ? (last_ - y) : R;
        const int32_t n = rows * W;
        const int32_t iter = (y - first_) / R;
        const int32_t slot = iter & 1;
        if (iter != 0) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        }
        LoadStripToL0B(y, n);
        if (iter >= 2) {
            if (slot == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID4);
            }
        }
        auto result = c_[slot * 16 * P];
        AscendC::MmadParams mm{};
        mm.m = 16;
        mm.n = static_cast<uint16_t>(n);
        mm.k = 96;
        mm.cmatrixInitVal = true;
        AscendC::Mmad(result, a0_, b0_, mm);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        if (iter != 0) {
            AscendC::CrossCoreWaitFlag<2>(9);
        }
        AscendC::FixpipeParamsV220 f{};
        f.mSize = 12;
        f.nSize = static_cast<uint16_t>(n);
        f.srcStride = 16;
        f.dstStride = P;
        AscendC::Fixpipe(ring_, result, f);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(2);
        if (slot == 0) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID4);
        }
    }

    __aicore__ inline void FinishSync()
    {
        if (last_ > first_) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
        }
        if (last_ > first_ + R) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID4);
        }
        AscendC::CrossCoreWaitFlag<2>(9);
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::A1> al_;
    AscendC::TBuf<AscendC::TPosition::B1> bl_;
    AscendC::TBuf<AscendC::TPosition::A2> a0b_;
    AscendC::TBuf<AscendC::TPosition::B2> b0b_;
    AscendC::TBuf<AscendC::TPosition::CO1> cb_;
    AscendC::LocalTensor<float> a1_;
    AscendC::LocalTensor<float> b1_;
    AscendC::LocalTensor<float> a0_;
    AscendC::LocalTensor<float> b0_;
    AscendC::LocalTensor<float> c_;
    AscendC::GlobalTensor<float> fg_;
    AscendC::GlobalTensor<float> ring_;
    int32_t batch_ = 0;
    int32_t first_ = 0;
    int32_t last_ = 0;
};

template <int32_t W, int32_t R>
class DirectProjectionVector : protected FdkProjectionVectorBase {
    static constexpr int32_t P = R * W;
    static constexpr int32_t MAX = 6912;

public:
    __aicore__ inline void Run(
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &td)
    {
        InitProjectionState(weight, bias, scratch, td, R, P, MAX);
        PackProjectionWeights(raw_, temp_, ix_, wg_, ring_, sub_);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
        BuildGatherIndex();
        ConsumeTiles();
        Decode();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void BuildGatherIndex()
    {
        auto seq = coord_.ReinterpretCast<int32_t>();
        FillSequence(seq, P);
        auto seqU32 = seq.ReinterpretCast<uint32_t>();
        AscendC::ShiftLeft(seqU32, seqU32, static_cast<uint32_t>(2), P);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t k = 0; k < 2; ++k) {
            for (int32_t tap = 0; tap < 3; ++tap) {
                const int32_t plane = k * 3 + tap;
                const int32_t offset = (8 + plane * P + tap - 1) * 4;
                AscendC::Adds(ix_[plane * P].ReinterpretCast<int32_t>(), seq, offset, P);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (int32_t k = 0; k < 2; ++k) {
            for (int32_t row = 0; row < R; ++row) {
                ix_.SetValue(k * 3 * P + row * W, 0U);
                ix_.SetValue((k * 3 + 2) * P + row * W + W - 1, 0U);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline void ConsumeTiles()
    {
        for (int32_t y = first_; y < last_; y += R) {
            const int32_t count = (last_ - y < R) ? (last_ - y) : R;
            const int32_t iter = (y - first_) / R;
            const int32_t slot = iter % 2;
            auto tileRaw = rawBuf_.Get<float>()[slot * (6 * P + 16)];
            raw_ = tileRaw[8];
            AscendC::CrossCoreWaitFlag<2>(2);
            if (iter >= 2) {
                if (slot == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                }
            }
            AscendC::DataCopy(raw_, ring_[sub_ * 6 * P], 6 * P);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(9);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Duplicate(tileRaw, 0.0f, 8);
            AscendC::Duplicate(raw_[6 * P], 0.0f, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(temp_, tileRaw, ix_, 0, 6 * P);
            AscendC::PipeBarrier<PIPE_V>();
            AccumulateChannelPair(y - first_, count, W, P, MAX);
            if (slot == 0) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void Decode()
    {
        const int32_t count = rows_ * W;
        AscendC::DataCopyPad(
            small_[64], bg_, AscendC::DataCopyExtParams{1, 16, 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        FillSequence(coord_, W);
        FillSequence(coord_[128], rows_);
        AscendC::Adds(coord_[128], coord_[128], static_cast<float>(first_), rows_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        for (int32_t k = 0; k < 2; ++k) {
            DecodeChannel(k, count);
        }
    }

    __aicore__ inline void DecodeChannel(int32_t k, int32_t count)
    {
        auto z = acc_[k * MAX];
        const float biasValue = small_.GetValue(64 + sub_ * 2 + k);
        const float peak = ExpShiftInPlace(z, temp_, small_, count, t_.invTau);
        AscendC::WholeReduceSum(coord_[256], z, static_cast<uint64_t>(64), static_cast<uint8_t>(rows_), 1, 1, W / 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[8], coord_[256], temp_, rows_);
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams xp{
            1, 1, 1, static_cast<uint8_t>(W / 8), static_cast<uint8_t>(W / 8), 0};
        AscendC::Mul(raw_, z, coord_, static_cast<uint64_t>(64), static_cast<uint8_t>(rows_), xp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[16], raw_, temp_, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(coord_[320], coord_[256], coord_[128], rows_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[24], coord_[320], temp_, rows_);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        StoreChannelStats(k, peak + biasValue, small_.GetValue(8), small_.GetValue(16), small_.GetValue(24));
    }
};

__aicore__ inline void ReduceDirectWorkerChannel(
    AscendC::LocalTensor<float> raw,
    AscendC::LocalTensor<uint32_t> offU32,
    AscendC::LocalTensor<float> value,
    AscendC::LocalTensor<float> small,
    AscendC::GlobalTensor<float> kg,
    AscendC::GlobalTensor<float> sg,
    int32_t batch,
    int32_t c,
    int32_t parts,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    GatherStatLanes(raw, offU32, value, value[64], value[128], value[192], c, parts, 64);
    const StatMergeResult res = ReduceMergedStats(
        value, value[64], value[128], value[192], value[256], value[320], small, 64, t.invTau);
    const float xCoord = (SafeDiv(res.sumX, res.den) + HALF_PIXEL_OFFSET) * t.stride;
    const float yCoord = (SafeDiv(res.sumY, res.den) + HALF_PIXEL_OFFSET) * t.stride;
    StoreKeypointScore(small, kg, sg, batch * 4 + c, xCoord, yCoord, res.peak);
}

__aicore__ inline void MergeDirectProjection(
    GM_ADDR keypoints,
    GM_ADDR scores,
    GM_ADDR scratch,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    const int32_t worker = static_cast<int32_t>(AscendC::GetBlockIdx());
    if (worker >= static_cast<int32_t>(t.batch) * 4) {
        return;
    }
    const int32_t batch = worker / 4;
    const int32_t c = worker % 4;
    const int32_t parts = static_cast<int32_t>(t.groupsPerBatch);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valuesBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> smallBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherBuf;
    pipe.InitBuffer(rawBuf, 16 * 32 * 4);
    pipe.InitBuffer(valuesBuf, 6 * 64 * 4);
    pipe.InitBuffer(smallBuf, 64 * 4);
    pipe.InitBuffer(gatherBuf, 64 * 4);
    auto offsets = gatherBuf.Get<int32_t>();
    FillSequence(offsets, 64);
    AscendC::Muls(offsets, offsets, static_cast<int32_t>(128), 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::GlobalTensor<float> stats;
    AscendC::GlobalTensor<float> kg;
    AscendC::GlobalTensor<float> sg;
    const int32_t statsOff = static_cast<int32_t>(t.batch) * parts * static_cast<int32_t>(t.groupFloats);
    stats.SetGlobalBuffer((__gm__ float *)scratch + statsOff);
    kg.SetGlobalBuffer((__gm__ float *)keypoints);
    sg.SetGlobalBuffer((__gm__ float *)scores);
    auto raw = rawBuf.Get<float>();
    AscendC::DataCopy(raw, stats[batch * parts * 32], parts * 32);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    ReduceDirectWorkerChannel(
        raw, offsets.ReinterpretCast<uint32_t>(), valuesBuf.Get<float>(),
        smallBuf.Get<float>(), kg, sg, batch, c, parts, t);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <int32_t PARTROWS>
__aicore__ inline void RunDirectProjection(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    GM_ADDR scratch = AscendC::GetSysWorkSpacePtr() + t.workspaceOffset;
    if ASCEND_IS_AIC {
        DirectProjectionCube<64, 2, PARTROWS> cubeOp;
        cubeOp.Run(feat, scratch, t);
    }
    if ASCEND_IS_AIV {
        DirectProjectionVector<64, 2> vecOp;
        vecOp.Run(weight, bias, scratch, t);
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        MergeDirectProjection(keypoints, scores, scratch, t);
    }
}
}  // namespace fdk

#endif  // FDK_DIRECT_PROJECTION_H
