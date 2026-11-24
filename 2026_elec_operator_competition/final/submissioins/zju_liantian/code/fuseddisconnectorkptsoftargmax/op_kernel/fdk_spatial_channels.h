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

#ifndef FDK_SPATIAL_CHANNELS_H
#define FDK_SPATIAL_CHANNELS_H

#include "fdk_common.h"

namespace fdk {
// Three-chain C8 spatial pipeline for B=1 72x96: two outputs share packed C8 input.
template <int32_t R, int32_t W>
class FdkSpatialChannels {
    static constexpr int32_t RAW_PLANE = (R + 2) * W;
    static constexpr int32_t N = R * W;
    static constexpr int32_t CHAINS = 3;
    static constexpr int32_t RAW = RAW_PLANE * 8;
    static constexpr int32_t LANES = 2 * CHAINS * N * 8;
    static constexpr int32_t TOTAL = 2 * RAW + LANES + 3 * 576 + 3 * N + 512 + 16;
    static_assert(TOTAL * 4 <= 196352, "C8 dual pipeline exceeds UB");

public:
    __aicore__ inline void Run(
        GM_ADDR feat,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR keypoints,
        GM_ADDR scores,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        InitBuffers(feat, weight, bias, keypoints, scores);
        const int32_t totalTasks = static_cast<int32_t>(t.batch) * 2 * static_cast<int32_t>(t.parts);
        const int32_t blockNum = static_cast<int32_t>(AscendC::GetBlockNum());
        for (int32_t task = static_cast<int32_t>(AscendC::GetBlockIdx()); task < totalTasks; task += blockNum) {
            RunTask(task, t);
        }
        AscendC::SyncAll<true>();
        const int32_t totalPlanes = static_cast<int32_t>(t.batch) * 4;
        for (int32_t plane = static_cast<int32_t>(AscendC::GetBlockIdx()); plane < totalPlanes; plane += blockNum) {
            MergePlane(plane, static_cast<int32_t>(t.parts), t.invTau, t.stride);
        }
    }

private:
    __aicore__ inline void InitBuffers(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores)
    {
        x_.SetGlobalBuffer((__gm__ float *)feat);
        w_.SetGlobalBuffer((__gm__ float *)weight);
        b_.SetGlobalBuffer((__gm__ float *)bias);
        kg_.SetGlobalBuffer((__gm__ float *)keypoints);
        sg_.SetGlobalBuffer((__gm__ float *)scores);
        pg_.SetGlobalBuffer((__gm__ float *)AscendC::GetSysWorkSpacePtr());
        pipe_.InitBuffer(buf_, TOTAL * 4);
        raw_ = buf_.Get<float>();
        lanes_ = raw_[RAW];
        wr_ = lanes_[LANES];
        coeff_ = wr_[576];
        map_ = coeff_[576].ReinterpretCast<uint32_t>();
        heat_ = coeff_[1152];
        tmp_ = heat_[N];
        scratch_ = tmp_[N];
        small_ = scratch_[N];
        transposed_ = small_[512];
    }

    __aicore__ inline void RunTask(int32_t task, const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        const int32_t parts = static_cast<int32_t>(t.parts);
        const int32_t part = task % parts;
        const int32_t group = task / parts;
        const int32_t batch = group / 2;
        const int32_t k0 = (group % 2) * 2;
        const int32_t y0 = part * R;
        const int32_t h = static_cast<int32_t>(t.height);
        AscendC::Duplicate(raw_, 0.0f, RAW);
        AscendC::Duplicate(lanes_, 0.0f, LANES);
        AscendC::Duplicate(transposed_, 0.0f, 8);
        AscendC::Duplicate(transposed_[RAW + 8], 0.0f, 8);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopy(wr_, w_[k0 * 288], 576);
        LoadChunk(batch, 0, h, y0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        PrepareWeightCoeffs();
        ConvolveAllGroups(batch, h, y0);
        AscendC::CreateVecIndex(small_[128], 0.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(small_[192], small_[128], 64.0f, W - 64);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t out = 0; out < 2; ++out) {
            DecodeOutput(out, batch, k0, part, y0, parts, t.invTau);
        }
    }

    __aicore__ inline void PrepareWeightCoeffs()
    {
        auto q = map_[80];
        auto lane = map_[160];
        auto mapI32 = map_.ReinterpretCast<int32_t>();
        auto laneI32 = lane.ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(mapI32, 0, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(map_[64].ReinterpretCast<int32_t>(), mapI32, 64, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(q, map_, static_cast<uint32_t>(3), 72);
        AscendC::ShiftLeft(lane, map_, static_cast<uint32_t>(29), 72);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(lane, lane, static_cast<uint32_t>(29), 72);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(laneI32, laneI32, static_cast<int32_t>(9), 72);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(mapI32, q.ReinterpretCast<int32_t>(), laneI32, 72);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(map_, map_, static_cast<uint32_t>(2), 72);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t out = 0; out < 2; ++out) {
            for (int32_t cg = 0; cg < 4; ++cg) {
                const uint32_t off = static_cast<uint32_t>((out * 288 + cg * 72) * 4);
                AscendC::Gather(coeff_[(out * 4 + cg) * 72], wr_, map_, off, 72);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ConvolveAllGroups(int32_t batch, int32_t h, int32_t y0)
    {
        const AscendC::BinaryRepeatParams rep{1, 1, 0, 8, 8, 0};
        for (int32_t cg = 0; cg < 4; ++cg) {
            if (cg > 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            TransposeChunk(transposed_[8]);
            AscendC::PipeBarrier<PIPE_V>();
            if (cg < 3) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                LoadChunk(batch, cg + 1, h, y0);
            }
            for (int32_t dy = 0; dy < 3; ++dy) {
                for (int32_t dx = 0; dx < 3; ++dx) {
                    for (int32_t out = 0; out < 2; ++out) {
                        const int32_t dstOff = (out * CHAINS + dx) * 8 * N;
                        const int32_t srcOff = (dy * W + dx) * 8;
                        const int32_t wOff = ((out * 4 + cg) * 9 + dy * 3 + dx) * 8;
                        AscendC::MulAddDst(
                            lanes_[dstOff], transposed_[srcOff], coeff_[wOff],
                            static_cast<uint64_t>(64), static_cast<uint8_t>(N / 8), rep);
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void DecodeOutput(
        int32_t out,
        int32_t batch,
        int32_t k0,
        int32_t part,
        int32_t y0,
        int32_t parts,
        float invTau)
    {
        const int32_t baseLane = out * CHAINS * 8 * N;
        AscendC::Duplicate(lanes_[baseLane], 0.0f, static_cast<uint64_t>(8), R, 1, W);
        AscendC::Duplicate(lanes_[baseLane + 2 * 8 * N + (W - 1) * 8], 0.0f, static_cast<uint64_t>(8), R, 1, W);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(lanes_[baseLane], lanes_[baseLane], lanes_[baseLane + 8 * N], 8 * N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(lanes_[baseLane], lanes_[baseLane], lanes_[baseLane + 2 * 8 * N], 8 * N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum(heat_, lanes_[baseLane], N / 8, 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax(
            small_, heat_, static_cast<uint64_t>(64), R, 1, 1, W / 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::WholeReduceMax(
            small_[256], heat_[64], static_cast<uint64_t>(W - 64), R, 1, 1, W / 8,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(small_, small_, small_[256], R);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax(small_[32], small_, scratch_, R, false);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float mx = small_.GetValue(32);
        ComputeStripMoments(out, batch, k0, part, y0, parts, invTau, mx);
    }

    __aicore__ inline void WriteStripStats(
        int32_t out,
        int32_t batch,
        int32_t k0,
        int32_t part,
        int32_t y0,
        int32_t parts,
        float mx)
    {
        float den = 0.0f;
        float sx = 0.0f;
        float sy = 0.0f;
        for (int32_t r = 0; r < R; ++r) {
            const float d = small_.GetValue(r);
            den += d;
            sx += small_.GetValue(32 + r);
            sy += static_cast<float>(y0 + r) * d;
        }
        small_.SetValue(64, mx);
        small_.SetValue(65, den);
        small_.SetValue(66, sx);
        small_.SetValue(67, sy);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(pg_[((batch * 4 + k0 + out) * parts + part) * 8], small_[64], 8);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeStripMoments(
        int32_t out,
        int32_t batch,
        int32_t k0,
        int32_t part,
        int32_t y0,
        int32_t parts,
        float invTau,
        float mx)
    {
        AscendC::Adds(heat_, heat_, -mx, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(heat_, heat_, invTau, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(heat_, heat_, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(small_, heat_, static_cast<uint64_t>(64), R, 1, 1, W / 8);
        AscendC::WholeReduceSum(small_[256], heat_[64], static_cast<uint64_t>(W - 64), R, 1, 1, W / 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(small_, small_, small_[256], R);
        const AscendC::BinaryRepeatParams xp{
            1, 1, 1, static_cast<uint8_t>(W / 8), static_cast<uint8_t>(W / 8), 0};
        AscendC::Mul(tmp_, heat_, small_[128], static_cast<uint64_t>(64), R, xp);
        AscendC::Mul(tmp_[64], heat_[64], small_[192], static_cast<uint64_t>(W - 64), R, xp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(small_[32], tmp_, static_cast<uint64_t>(64), R, 1, 1, W / 8);
        AscendC::WholeReduceSum(small_[256], tmp_[64], static_cast<uint64_t>(W - 64), R, 1, 1, W / 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(small_[32], small_[32], small_[256], R);
        AscendC::PipeBarrier<PIPE_ALL>();
        WriteStripStats(out, batch, k0, part, y0, parts, mx);
    }

    __aicore__ inline void LoadChunk(int32_t batch, int32_t cg, int32_t h, int32_t y0)
    {
        const int32_t first = (y0 > 0) ? (y0 - 1) : 0;
        const int32_t last = (y0 + R + 1 < h) ? (y0 + R + 1) : h;
        const int32_t count = (last - first) * W;
        const int32_t offset = (first - y0 + 1) * W;
        const int32_t full = h * W;
        const AscendC::DataCopyParams cp{
            8,
            static_cast<uint16_t>(count / 8),
            static_cast<uint16_t>((full - count) / 8),
            static_cast<uint16_t>((RAW_PLANE - count) / 8)};
        AscendC::DataCopy(raw_[offset], x_[(batch * 32 + cg * 8) * full + first * W], cp);
    }

    __aicore__ inline void TransposeChunk(AscendC::LocalTensor<float> dstTensor)
    {
        uint64_t src[16];
        uint64_t dst[16];
        for (int32_t i = 0; i < 16; ++i) {
            src[i] = static_cast<uint64_t>(raw_[(i % 8) * RAW_PLANE + (i / 8) * 8].GetPhyAddr());
            dst[i] = static_cast<uint64_t>(dstTensor[((i / 2) + (i % 2) * 8) * 8].GetPhyAddr());
        }
        const AscendC::TransDataTo5HDParams tp{false, false, RAW_PLANE / 16, 16, 2};
        AscendC::TransDataTo5HD<float>(dst, src, tp);
    }

    __aicore__ inline void MergePlane(int32_t plane, int32_t parts, float invTau, float stride)
    {
        auto m = raw_[512];
        auto den = m[64];
        auto x = den[64];
        auto y = x[64];
        auto tmp = y[64];
        auto out = tmp[64];
        auto scratch = out[64];
        auto ix = scratch[64].ReinterpretCast<uint32_t>();
        AscendC::DataCopy(raw_, pg_[plane * parts * 8], parts * 8);
        AscendC::DataCopyPad(
            out[32], b_[plane % 4], AscendC::DataCopyExtParams{1, 4, 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        AscendC::CreateVecIndex(tmp, 0.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmp, tmp, 32.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ix.ReinterpretCast<int32_t>(), tmp, AscendC::RoundMode::CAST_RINT, 64);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Gather(m, raw_, ix, 0, parts);
        AscendC::Gather(den, raw_, ix, 4, parts);
        AscendC::Gather(x, raw_, ix, 8, parts);
        AscendC::Gather(y, raw_, ix, 12, parts);
        AscendC::PipeBarrier<PIPE_V>();
        FinalizeMergedPlane(m, den, x, y, out, scratch, plane, parts, invTau, stride);
    }

    __aicore__ inline void FinalizeMergedPlane(
        AscendC::LocalTensor<float> m,
        AscendC::LocalTensor<float> den,
        AscendC::LocalTensor<float> x,
        AscendC::LocalTensor<float> y,
        AscendC::LocalTensor<float> out,
        AscendC::LocalTensor<float> scratch,
        int32_t plane,
        int32_t parts,
        float invTau,
        float stride)
    {
        AscendC::ReduceMax(out, m, scratch, parts, false);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float maximum = out.GetValue(0);
        AscendC::Adds(m, m, -maximum, parts);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(m, m, invTau, parts);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(m, m, parts);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(den, den, m, parts);
        AscendC::Mul(x, x, m, parts);
        AscendC::Mul(y, y, m, parts);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(out, den, scratch, parts);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(out[8], x, scratch, parts);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(out[16], y, scratch, parts);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float d = out.GetValue(0);
        const float sx = out.GetValue(8);
        const float sy = out.GetValue(16);
        const float bVal = out.GetValue(32);
        out.SetValue(0, (SafeDiv(sx, d) + HALF_PIXEL_OFFSET) * stride);
        out.SetValue(1, (SafeDiv(sy, d) + HALF_PIXEL_OFFSET) * stride);
        out.SetValue(8, maximum + bVal);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(kg_[plane * 2], out, AscendC::DataCopyExtParams{1, 8, 0, 0, 0});
        AscendC::DataCopyPad(sg_[plane], out[8], AscendC::DataCopyExtParams{1, 4, 0, 0, 0});
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_;
    AscendC::LocalTensor<float> raw_;
    AscendC::LocalTensor<float> lanes_;
    AscendC::LocalTensor<float> wr_;
    AscendC::LocalTensor<float> coeff_;
    AscendC::LocalTensor<uint32_t> map_;
    AscendC::LocalTensor<float> heat_;
    AscendC::LocalTensor<float> tmp_;
    AscendC::LocalTensor<float> scratch_;
    AscendC::LocalTensor<float> small_;
    AscendC::LocalTensor<float> transposed_;
    AscendC::GlobalTensor<float> x_;
    AscendC::GlobalTensor<float> w_;
    AscendC::GlobalTensor<float> b_;
    AscendC::GlobalTensor<float> kg_;
    AscendC::GlobalTensor<float> sg_;
    AscendC::GlobalTensor<float> pg_;
};

__aicore__ inline void RunSpatialChannels(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    FdkSpatialChannels<4, 96> op;
    op.Run(feat, weight, bias, keypoints, scores, t);
}
}  // namespace fdk

#endif  // FDK_SPATIAL_CHANNELS_H
