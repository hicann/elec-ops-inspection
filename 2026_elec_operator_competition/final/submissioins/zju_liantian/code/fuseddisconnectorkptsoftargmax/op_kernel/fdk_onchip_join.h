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

#ifndef FDK_ONCHIP_JOIN_H
#define FDK_ONCHIP_JOIN_H

#include "fdk_fallback.h"
#include "fdk_onchip_join_cube.h"

namespace fdk {
template <int32_t W, int32_t ROWS>
class FdkOnchipJoin {
    static constexpr int32_t N = W * ROWS;
    static constexpr int32_t MAXS = 6912;

public:
    __aicore__ inline void Run(
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        InitBuffers(weight, bias, scratch, t);
        if (group_ == 0) {
            PackWeights();
        }
        AscendC::SyncAll<false>();
        AccumulateTiles();
        Decode();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void InitBuffers(
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        h_ = static_cast<int32_t>(t.height);
        invTau_ = t.invTau;
        group_ = static_cast<int32_t>(AscendC::GetBlockIdx()) / 2;
        sub_ = static_cast<int32_t>(AscendC::GetSubBlockIdx());
        const int32_t gpb = static_cast<int32_t>(t.groupsPerBatch);
        const int32_t tiles = (h_ + ROWS - 1) / ROWS;
        const int32_t partTiles = (tiles + gpb - 1) / gpb;
        firstRow_ = (group_ % gpb) * partTiles * ROWS;
        const int32_t candLast = firstRow_ + partTiles * ROWS;
        lastRow_ = (candLast > h_) ? h_ : candLast;
        localRows_ = lastRow_ - firstRow_;
        co_ = sub_ * 2;
        wg_.SetGlobalBuffer((__gm__ float *)weight);
        bg_.SetGlobalBuffer((__gm__ float *)bias);
        const int32_t gf = static_cast<int32_t>(t.groupFloats);
        const int32_t totalGroups = static_cast<int32_t>(t.batch) * gpb;
        ring_.SetGlobalBuffer((__gm__ float *)scratch + 1408 + group_ * gf);
        packed_.SetGlobalBuffer((__gm__ float *)scratch);
        stats_.SetGlobalBuffer((__gm__ float *)scratch + 1408 + totalGroups * gf);
        InitLocalTensors();
    }

    __aicore__ inline void InitLocalTensors()
    {
        const int32_t idxElems = (3 * N > 1152) ? (3 * N) : 1152;
        pipe_.InitBuffer(inBuf_, MAXS * 4);
        pipe_.InitBuffer(outBuf_, MAXS * 4);
        pipe_.InitBuffer(metaBuf_, 1024 * 4);
        pipe_.InitBuffer(indexBuf_, idxElems * 4);
        pipe_.InitBuffer(heatBuf_, (12 * N + 48) * 4);
        pipe_.InitBuffer(accBuf_, 2 * MAXS * 4);
        pipe_.InitBuffer(smallBuf_, 64 * 4);
        input_ = inBuf_.Get<float>();
        output_ = outBuf_.Get<float>();
        meta_ = metaBuf_.Get<float>();
        index_ = indexBuf_.Get<int32_t>();
        heat_ = heatBuf_.Get<float>();
        acc_ = accBuf_.Get<float>();
        small_ = smallBuf_.Get<float>();
    }

    __aicore__ inline void PackWeights()
    {
        FillSequence(index_, 32);
        AscendC::Muls(index_, index_, static_cast<int32_t>(9), 32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(index_[32], index_, static_cast<int32_t>(3), 32);
        AscendC::Adds(index_[64], index_, static_cast<int32_t>(6), 32);
        AscendC::PipeBarrier<PIPE_V>();
        auto idxU32 = index_.ReinterpretCast<uint32_t>();
        AscendC::ShiftLeft(idxU32, idxU32, static_cast<uint32_t>(2), 96);
        AscendC::DataCopy(input_, wg_[sub_ * 576], 576);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        for (int32_t dx = 0; dx < 3; ++dx) {
            for (int32_t c = 0; c < 2; ++c) {
                AscendC::Gather(output_, input_[c * 288], idxU32, static_cast<uint32_t>(dx * 4), 96);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(packed_[(dx * 4 + co_ + c) * 96], output_, 96);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        if (sub_ == 0) {
            PackJoinConstants();
        }
    }

    __aicore__ inline void PackJoinConstants()
    {
        auto coeff = output_.ReinterpretCast<half>();
        AscendC::Duplicate(coeff, static_cast<half>(0), 512);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (int32_t c = 0; c < 16; ++c) {
            coeff.SetValue(c * 16 + c, static_cast<half>(-1));
        }
        for (int32_t c = 0; c < 4; ++c) {
            for (int32_t dx = 0; dx < 3; ++dx) {
                coeff.SetValue(256 + c * 48 + dx * 16 + dx * 4 + c, static_cast<half>(1));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopy(packed_[1152], output_, 256);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void AccumulateTiles()
    {
        FillSequence(index_, N);
        AscendC::Muls(index_, index_, static_cast<int32_t>(16), N);
        AscendC::PipeBarrier<PIPE_V>();
        auto idxU32 = index_.ReinterpretCast<uint32_t>();
        for (int32_t y = firstRow_; y < lastRow_; y += ROWS) {
            AscendC::CrossCoreWaitFlag<2>(2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::DataCopy(heat_, ring_, 4 * N);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(9);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            for (int32_t c = 0; c < 2; ++c) {
                auto dst = acc_[c * MAXS + (y - firstRow_) * W];
                AscendC::Gather(dst, heat_, idxU32, static_cast<uint32_t>((co_ + c) * 4), N);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void Decode()
    {
        const int32_t count = localRows_ * W;
        AscendC::DataCopyPad(
            small_[40], bg_, AscendC::DataCopyExtParams{1, 16, 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        FillSequence(meta_, W);
        for (int32_t row = 0; row < localRows_; ++row) {
            AscendC::Duplicate(meta_[128 + row * 8], static_cast<float>(row + firstRow_), 8);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        for (int32_t c = 0; c < 2; ++c) {
            DecodeChannel(c, count);
        }
    }

    __aicore__ inline void DecodeChannel(int32_t c, int32_t count)
    {
        auto values = acc_[c * MAXS];
        AscendC::Adds(values, values, small_.GetValue(40 + co_ + c), count);
        AscendC::PipeBarrier<PIPE_V>();
        const float peak = ExpShiftInPlace(values, output_, small_, count, invTau_);
        AscendC::ReduceSum(small_[8], values, output_, count);
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams xp{
            1, 1, 1, static_cast<uint8_t>(W / 8), static_cast<uint8_t>(W / 8), 0};
        AscendC::Mul(input_, values, meta_, static_cast<uint64_t>(64), static_cast<uint8_t>(localRows_), xp);
        if constexpr (W > 64) {
            AscendC::Mul(
                input_[64], values[64], meta_[64],
                static_cast<uint64_t>(W - 64), static_cast<uint8_t>(localRows_), xp);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[16], input_, output_, count);
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams yp{
            1, 1, 0, static_cast<uint8_t>(W / 8), static_cast<uint8_t>(W / 8), 1};
        AscendC::Mul(input_, values, meta_[128], static_cast<uint64_t>(64), static_cast<uint8_t>(localRows_), yp);
        if constexpr (W > 64) {
            AscendC::Mul(
                input_[64], values[64], meta_[128],
                static_cast<uint64_t>(W - 64), static_cast<uint8_t>(localRows_), yp);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[24], input_, output_, count);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        WriteChannelStats(c, peak);
    }

    __aicore__ inline void WriteChannelStats(int32_t c, float peak)
    {
        small_.SetValue(32, peak);
        small_.SetValue(33, small_.GetValue(8));
        small_.SetValue(34, small_.GetValue(16));
        small_.SetValue(35, small_.GetValue(24));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopy(stats_[group_ * 32 + (co_ + c) * 8], small_[32], 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> metaBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> heatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> smallBuf_;
    AscendC::LocalTensor<float> input_;
    AscendC::LocalTensor<float> output_;
    AscendC::LocalTensor<float> meta_;
    AscendC::LocalTensor<float> heat_;
    AscendC::LocalTensor<float> acc_;
    AscendC::LocalTensor<float> small_;
    AscendC::LocalTensor<int32_t> index_;
    AscendC::GlobalTensor<float> wg_;
    AscendC::GlobalTensor<float> bg_;
    AscendC::GlobalTensor<float> ring_;
    AscendC::GlobalTensor<float> stats_;
    AscendC::GlobalTensor<float> packed_;
    int32_t h_ = 0;
    int32_t group_ = 0;
    int32_t sub_ = 0;
    int32_t co_ = 0;
    int32_t firstRow_ = 0;
    int32_t lastRow_ = 0;
    int32_t localRows_ = 0;
    float invTau_ = 1.0f;
};

__aicore__ inline uint32_t ReduceJoinChannels(
    AscendC::LocalTensor<float> raw,
    AscendC::LocalTensor<uint32_t> gatherIdx,
    AscendC::LocalTensor<float> value,
    AscendC::LocalTensor<float> small,
    AscendC::GlobalTensor<float> kg,
    AscendC::GlobalTensor<float> sg,
    int32_t batch,
    int32_t co,
    int32_t parts,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    uint32_t invalid = 0U;
    for (int32_t c = co; c < co + 2; ++c) {
        GatherStatLanes(raw, gatherIdx, value, value[64], value[128], value[192], c, parts, 64);
        const StatMergeResult res = ReduceMergedStats(
            value, value[64], value[128], value[192], value[256], value[320], small, 64, t.invTau);
        const bool peakBad = !(res.peak <= POS_MAX_F32 && res.peak >= NEG_MAX_F32);
        if ((t.useJoin != 0U) && (!(res.den > 0.0f) || peakBad)) {
            invalid |= (static_cast<uint32_t>(1) << static_cast<uint32_t>(c - co));
            continue;
        }
        const float xCoord = (SafeDiv(res.sumX, res.den) + HALF_PIXEL_OFFSET) * t.stride;
        const float yCoord = (SafeDiv(res.sumY, res.den) + HALF_PIXEL_OFFSET) * t.stride;
        StoreKeypointScore(small, kg, sg, batch * 4 + c, xCoord, yCoord, res.peak);
    }
    return invalid;
}

__aicore__ inline uint32_t MergeJoinStats(
    GM_ADDR keypoints,
    GM_ADDR scores,
    GM_ADDR scratch,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    const int32_t group = static_cast<int32_t>(AscendC::GetBlockIdx()) / 2;
    const int32_t parts = static_cast<int32_t>(t.groupsPerBatch);
    if (group % parts != 0) {
        return 0U;
    }
    const int32_t batch = group / parts;
    const int32_t co = static_cast<int32_t>(AscendC::GetSubBlockIdx()) * 2;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valuesBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> smallBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherBuf;
    pipe.InitBuffer(rawBuf, 16 * 32 * 4);
    pipe.InitBuffer(valuesBuf, 6 * 64 * 4);
    pipe.InitBuffer(smallBuf, 64 * 4);
    pipe.InitBuffer(gatherBuf, 64 * 4);
    auto gather = gatherBuf.Get<int32_t>();
    FillSequence(gather, 64);
    AscendC::Muls(gather, gather, static_cast<int32_t>(128), 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::GlobalTensor<float> stats;
    AscendC::GlobalTensor<float> kg;
    AscendC::GlobalTensor<float> sg;
    const int32_t statsOff = 1408 + static_cast<int32_t>(t.batch) * parts * static_cast<int32_t>(t.groupFloats);
    stats.SetGlobalBuffer((__gm__ float *)scratch + statsOff);
    kg.SetGlobalBuffer((__gm__ float *)keypoints);
    sg.SetGlobalBuffer((__gm__ float *)scores);
    auto raw = rawBuf.Get<float>();
    AscendC::DataCopy(raw, stats[batch * parts * 32], parts * 32);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    const uint32_t invalid = ReduceJoinChannels(
        raw, gather.ReinterpretCast<uint32_t>(), valuesBuf.Get<float>(),
        smallBuf.Get<float>(), kg, sg, batch, co, parts, t);
    AscendC::PipeBarrier<PIPE_ALL>();
    return invalid;
}

template <int32_t W, int32_t ROWS>
__aicore__ inline void RunOnchipJoin(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    GM_ADDR scratch = AscendC::GetSysWorkSpacePtr() + t.workspaceOffset;
    if ASCEND_IS_AIC {
        FdkOnchipJoinCube<W, ROWS> cubeOp;
        cubeOp.Run(feat, scratch, t);
    }
    if ASCEND_IS_AIV {
        FdkOnchipJoin<W, ROWS> vecOp;
        vecOp.Run(weight, bias, scratch, t);
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        const uint32_t invalid = MergeJoinStats(keypoints, scores, scratch, t);
        if (invalid != 0U) {
            const int32_t group = static_cast<int32_t>(AscendC::GetBlockIdx()) / 2;
            const int32_t batch = group / static_cast<int32_t>(t.groupsPerBatch);
            const int32_t base = batch * 4 + static_cast<int32_t>(AscendC::GetSubBlockIdx()) * 2;
            for (int32_t c = 0; c < 2; ++c) {
                if ((invalid & (static_cast<uint32_t>(1) << static_cast<uint32_t>(c))) != 0U) {
                    RunFallbackPlane(feat, weight, bias, keypoints, scores, t, base + c);
                }
            }
        }
    }
}
}  // namespace fdk

#endif  // FDK_ONCHIP_JOIN_H
