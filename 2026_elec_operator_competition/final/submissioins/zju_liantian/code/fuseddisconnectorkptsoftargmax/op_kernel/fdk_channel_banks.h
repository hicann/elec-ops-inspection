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

#ifndef FDK_CHANNEL_BANKS_H
#define FDK_CHANNEL_BANKS_H

#include "fdk_common.h"

namespace fdk {
// AIV channel-bank cache path for 3x48x64: 36 tasks (12 planes x 3 row strips).
class FdkChannelBanks {
    static constexpr int32_t W = 64;
    static constexpr int32_t R = 16;
    static constexpr int32_t N = R * W;
    static constexpr int32_t HALO = (R + 2) * W;
    static constexpr int32_t INPUT = 32 * HALO + 8;

public:
    __aicore__ inline void Run(
        GM_ADDR feat,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        InitAndLoad(feat, weight, bias, scratch);
        PrepareShiftIndices();
        ConvolveAndCombine();
        DecodeAndWriteStats(t);
    }

private:
    __aicore__ inline void InitBuffers(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR scratch)
    {
        fg_.SetGlobalBuffer((__gm__ float *)feat);
        wg_.SetGlobalBuffer((__gm__ float *)weight);
        bg_.SetGlobalBuffer((__gm__ float *)bias);
        st_.SetGlobalBuffer((__gm__ float *)scratch);
        pipe_.InitBuffer(inputBuf_, INPUT * 4);
        pipe_.InitBuffer(weightBuf_, 288 * 4);
        pipe_.InitBuffer(accBuf_, (3 * N + 8) * 4);
        pipe_.InitBuffer(shiftBuf_, HALO * 4);
        pipe_.InitBuffer(tempBuf_, N * 4);
        pipe_.InitBuffer(indexBuf_, 2 * HALO * 4);
        pipe_.InitBuffer(metaBuf_, 512 * 4);
        pipe_.InitBuffer(smallBuf_, 64 * 4);
        input_ = inputBuf_.Get<float>();
        weights_ = weightBuf_.Get<float>();
        acc_ = accBuf_.Get<float>();
        shifted_ = shiftBuf_.Get<float>();
        temp_ = tempBuf_.Get<float>();
        index_ = indexBuf_.Get<int32_t>();
        meta_ = metaBuf_.Get<float>();
        small_ = smallBuf_.Get<float>();
    }

    __aicore__ inline void InitAndLoad(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR scratch)
    {
        task_ = static_cast<int32_t>(AscendC::GetBlockIdx());
        const int32_t plane = task_ / 3;
        const int32_t part = task_ % 3;
        const int32_t batch = plane / 4;
        const int32_t co = plane % 4;
        y_ = part * R;
        const int32_t first = (y_ > 0) ? (y_ - 1) : 0;
        const int32_t end = (y_ + R + 1 < 48) ? (y_ + R + 1) : 48;
        const int32_t valid = (end - first) * W;
        const int32_t top = (y_ == 0) ? W : 0;
        InitBuffers(feat, weight, bias, scratch);
        AscendC::Duplicate(input_, 0.0f, INPUT);
        AscendC::Duplicate(acc_, 0.0f, 3 * N + 8);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        const AscendC::DataCopyParams cp{
            32,
            static_cast<uint16_t>(valid / 8),
            static_cast<uint16_t>((3072 - valid) / 8),
            static_cast<uint16_t>((HALO - valid) / 8)};
        AscendC::DataCopy(input_[top], fg_[batch * 32 * 3072 + first * W], cp);
        AscendC::DataCopy(weights_, wg_[co * 288], 288);
        AscendC::DataCopyPad(
            small_[40], bg_[co], AscendC::DataCopyExtParams{1, 4, 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
    }

    __aicore__ inline void PrepareShiftIndices()
    {
        FillSequence(index_, HALO);
        AscendC::Adds(index_[HALO], index_, static_cast<int32_t>(1), HALO);
        AscendC::Adds(index_, index_, static_cast<int32_t>(-1), HALO);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        index_.SetValue(0, 0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        auto idxU32 = index_.ReinterpretCast<uint32_t>();
        AscendC::ShiftLeft(idxU32, idxU32, static_cast<uint32_t>(2), 2 * HALO);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ConvolveAndCombine()
    {
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>(0, 0xffffffffffffffffULL);
        for (int32_t ic = 0; ic < 32; ++ic) {
            TapRow<0>(ic);
            TapRow<1>(ic);
            TapRow<2>(ic);
        }
        AscendC::Gather(shifted_, acc_, index_.ReinterpretCast<uint32_t>(), 0, N);
        AscendC::Gather(temp_, acc_[2 * N], index_[HALO].ReinterpretCast<uint32_t>(), 0, N);
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t leftMask[2] = {1ULL, 0ULL};
        uint64_t rightMask[2] = {0x8000000000000000ULL, 0ULL};
        AscendC::Duplicate(shifted_, 0.0f, leftMask, static_cast<uint8_t>(R), 1, 8);
        AscendC::Duplicate(temp_, 0.0f, rightMask, static_cast<uint8_t>(R), 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc_[N], acc_[N], shifted_, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc_[N], acc_[N], temp_, N);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecodeAndWriteStats(const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        auto planeAcc = acc_[N];
        AscendC::ReduceMax(small_, planeAcc, temp_, N, false);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float peak = small_.GetValue(0);
        const float biasVal = small_.GetValue(40);
        AscendC::Adds(planeAcc, planeAcc, -peak, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(planeAcc, planeAcc, t.invTau, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(planeAcc, planeAcc, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[8], planeAcc, temp_, N);
        AscendC::PipeBarrier<PIPE_V>();
        FillSequence(meta_, 64);
        for (int32_t row = 0; row < R; ++row) {
            AscendC::Duplicate(meta_[64 + row * 8], static_cast<float>(y_ + row), 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(
            shifted_, planeAcc, meta_, static_cast<uint64_t>(64),
            static_cast<uint8_t>(R), AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 0});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[16], shifted_, temp_, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(
            shifted_, planeAcc, meta_[64], static_cast<uint64_t>(64),
            static_cast<uint8_t>(R), AscendC::BinaryRepeatParams{1, 1, 0, 8, 8, 1});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[24], shifted_, temp_, N);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        small_.SetValue(32, peak + biasVal);
        small_.SetValue(33, small_.GetValue(8));
        small_.SetValue(34, small_.GetValue(16));
        small_.SetValue(35, small_.GetValue(24));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopy(st_[task_ * 8], small_[32], 8);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <int32_t DY>
    __aicore__ inline void TapRow(int32_t ic)
    {
        auto src = input_[ic * HALO + DY * W];
        const float a = weights_.GetValue(ic * 9 + DY * 3);
        const float b = weights_.GetValue(ic * 9 + DY * 3 + 1);
        const float c = weights_.GetValue(ic * 9 + DY * 3 + 2);
        const AscendC::UnaryRepeatParams rp{1, 1, 8, 8};
        AscendC::Axpy<float, float, false>(acc_, src, a, static_cast<uint64_t>(64), static_cast<uint8_t>(R), rp);
        AscendC::Axpy<float, float, false>(acc_[N], src, b, static_cast<uint64_t>(64), static_cast<uint8_t>(R), rp);
        AscendC::Axpy<float, float, false>(acc_[2 * N], src, c, static_cast<uint64_t>(64), static_cast<uint8_t>(R), rp);
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> shiftBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> metaBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> smallBuf_;
    AscendC::LocalTensor<float> input_;
    AscendC::LocalTensor<float> weights_;
    AscendC::LocalTensor<float> acc_;
    AscendC::LocalTensor<float> shifted_;
    AscendC::LocalTensor<float> temp_;
    AscendC::LocalTensor<float> meta_;
    AscendC::LocalTensor<float> small_;
    AscendC::LocalTensor<int32_t> index_;
    AscendC::GlobalTensor<float> fg_;
    AscendC::GlobalTensor<float> wg_;
    AscendC::GlobalTensor<float> bg_;
    AscendC::GlobalTensor<float> st_;
    int32_t task_ = 0;
    int32_t y_ = 0;
};

__aicore__ inline void UnpackThreePartitions(AscendC::LocalTensor<float> mem)
{
    auto raw = mem;
    auto maxs = mem[64];
    auto mass = mem[128];
    auto xs = mem[192];
    auto ys = mem[256];
    AscendC::Duplicate(maxs, NEG_MAX_F32, 64);
    AscendC::Duplicate(mass, 0.0f, 64);
    AscendC::Duplicate(xs, 0.0f, 64);
    AscendC::Duplicate(ys, 0.0f, 64);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int32_t p = 0; p < 3; ++p) {
        maxs.SetValue(p, raw.GetValue(p * 8));
        mass.SetValue(p, raw.GetValue(p * 8 + 1));
        xs.SetValue(p, raw.GetValue(p * 8 + 2));
        ys.SetValue(p, raw.GetValue(p * 8 + 3));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
}

__aicore__ inline void MergeChannelBanks(
    GM_ADDR kp,
    GM_ADDR scores,
    GM_ADDR scratch,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    const int32_t task = static_cast<int32_t>(AscendC::GetBlockIdx());
    if (task % 3 != 0) {
        return;
    }
    const int32_t plane = task / 3;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 7 * 64 * 4);
    auto mem = buf.Get<float>();
    AscendC::GlobalTensor<float> st;
    AscendC::GlobalTensor<float> kg;
    AscendC::GlobalTensor<float> sg;
    st.SetGlobalBuffer((__gm__ float *)scratch);
    kg.SetGlobalBuffer((__gm__ float *)kp);
    sg.SetGlobalBuffer((__gm__ float *)scores);
    AscendC::DataCopy(mem, st[task * 8], 24);
    UnpackThreePartitions(mem);
    const StatMergeResult res = ReduceMergedStats(
        mem[64], mem[128], mem[192], mem[256], mem[320], mem, mem[384], 64, t.invTau);
    const float xCoord = (SafeDiv(res.sumX, res.den) + HALF_PIXEL_OFFSET) * t.stride;
    const float yCoord = (SafeDiv(res.sumY, res.den) + HALF_PIXEL_OFFSET) * t.stride;
    StoreKeypointScore(mem[384], kg, sg, plane, xCoord, yCoord, res.peak);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunChannelBanks(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    GM_ADDR scratch = AscendC::GetSysWorkSpacePtr() + t.workspaceOffset;
    FdkChannelBanks op;
    op.Run(feat, weight, bias, scratch, t);
    AscendC::SyncAll<true>();
    MergeChannelBanks(keypoints, scores, scratch, t);
}
}  // namespace fdk

#endif  // FDK_CHANNEL_BANKS_H
