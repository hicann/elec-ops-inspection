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

#ifndef FDK_VERTICAL_CUBE_H
#define FDK_VERTICAL_CUBE_H

#include "fdk_projection_vector_base.h"
#include "fdk_vertical_cube_stage.h"

namespace fdk {
template <int32_t W, int32_t R, int32_t MAX, bool DOUBLE_BUF, uint32_t FLAG_WT, uint32_t FLAG_OUT, uint32_t FLAG_ACK>
class FdkVerticalCube : protected FdkProjectionVectorBase {
    static constexpr int32_t P = R * W;

public:
    __aicore__ inline void Run(
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR scratch,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &td)
    {
        InitProjectionState(weight, bias, scratch, td, R, P, MAX);
        PackProjectionWeights(raw_, temp_, ix_, wg_, ring_, sub_);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_WT);
        BuildGatherIndex();
        ConsumeTiles();
        Decode();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void BuildGatherIndex()
    {
        auto ixI32 = ix_.ReinterpretCast<int32_t>();
        FillSequence(ixI32, P);
        for (int32_t plane = 5; plane >= 0; --plane) {
            const int32_t off = 8 + plane * P + (plane % 3) - 1;
            AscendC::Adds(ix_[plane * P].ReinterpretCast<int32_t>(), ixI32, off, P);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::ShiftLeft(ix_, ix_, static_cast<uint32_t>(2), 6 * P);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t plane = 0; plane < 6; ++plane) {
            for (int32_t row = 0; row < R; ++row) {
                if (plane % 3 == 0) {
                    ix_.SetValue(plane * P + row * W, 0U);
                }
                if (plane % 3 == 2) {
                    ix_.SetValue(plane * P + row * W + W - 1, 0U);
                }
            }
        }
        auto rawBase = rawBuf_.Get<float>();
        AscendC::Duplicate(rawBase, 0.0f, 8);
        AscendC::Duplicate(rawBase[6 * P + 16], 0.0f, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ConsumeTiles()
    {
        for (int32_t y = first_; y < last_; y += R) {
            const int32_t count = (last_ - y < R) ? (last_ - y) : R;
            const int32_t iter = (y - first_) / R;
            const int32_t slot = DOUBLE_BUF ? (iter % 2) : 0;
            auto tileRaw = rawBuf_.Get<float>()[slot * (6 * P + 16)];
            raw_ = tileRaw[8];
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(FLAG_OUT);
            if constexpr (DOUBLE_BUF) {
                if (iter >= 2) {
                    if (slot == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    } else {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                    }
                }
            }
            AscendC::DataCopy(raw_, ring_[sub_ * 6 * P], 6 * P);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(FLAG_ACK);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Gather(temp_, tileRaw, ix_, 0, 6 * P);
            AscendC::PipeBarrier<PIPE_V>();
            AccumulateChannelPair(y - first_, count, W, P, MAX);
            if constexpr (DOUBLE_BUF) {
                if (slot == 0) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                }
            } else {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        if constexpr (DOUBLE_BUF) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline void Decode()
    {
        const int32_t count = rows_ * W;
        AscendC::DataCopyPad(
            small_[64], bg_, AscendC::DataCopyExtParams{1, 16, 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        for (int32_t k = 0; k < 2; ++k) {
            DecodeChannel(k, count);
        }
    }

    __aicore__ inline void DecodeChannel(int32_t k, int32_t count)
    {
        auto z = acc_[k * MAX];
        AscendC::Adds(z, z, small_.GetValue(64 + sub_ * 2 + k), count);
        AscendC::PipeBarrier<PIPE_V>();
        const float peak = ExpShiftInPlace(z, temp_, small_, count, t_.invTau);
        AscendC::ReduceSum(small_[8], z, temp_, count);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float den = small_.GetValue(8);
        AccumulateMomentsAndStore(k, z, count, peak, den);
    }

    __aicore__ inline void AccumulateMomentsAndStore(
        int32_t k,
        AscendC::LocalTensor<float> z,
        int32_t count,
        float peak,
        float den)
    {
        FillSequence(coord_, W);
        for (int32_t r = 1; r < rows_; ++r) {
            AscendC::Adds(coord_[r * W], coord_, 0.0f, W);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(raw_, z, coord_, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[16], raw_, temp_, count);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float sx = small_.GetValue(16);
        for (int32_t r = 0; r < rows_; ++r) {
            AscendC::Duplicate(coord_[r * W], static_cast<float>(r + first_), W);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(raw_, z, coord_, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small_[24], raw_, temp_, count);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        StoreChannelStats(k, peak, den, sx, small_.GetValue(24));
    }
};

template <int32_t REDUCE_LEN>
__aicore__ inline void ReduceVerticalChannels(
    AscendC::LocalTensor<float> raw,
    AscendC::LocalTensor<uint32_t> ixU32,
    AscendC::LocalTensor<float> v,
    AscendC::LocalTensor<float> out,
    AscendC::GlobalTensor<float> kg,
    AscendC::GlobalTensor<float> sg,
    int32_t batch,
    int32_t co,
    int32_t parts,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    for (int32_t c = co; c < co + 2; ++c) {
        GatherStatLanes(raw, ixU32, v, v[64], v[128], v[192], c, parts, REDUCE_LEN);
        const StatMergeResult res = ReduceMergedStats(
            v, v[64], v[128], v[192], v[256], v[320], out, REDUCE_LEN, t.invTau);
        const float xCoord = (SafeDiv(res.sumX, res.den) + HALF_PIXEL_OFFSET) * t.stride;
        const float yCoord = (SafeDiv(res.sumY, res.den) + HALF_PIXEL_OFFSET) * t.stride;
        StoreKeypointScore(out, kg, sg, batch * 4 + c, xCoord, yCoord, res.peak);
    }
}

template <int32_t REDUCE_LEN>
__aicore__ inline void MergeVerticalCubeStats(
    GM_ADDR keypoints,
    GM_ADDR scores,
    GM_ADDR scratch,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    const int32_t group = static_cast<int32_t>(AscendC::GetBlockIdx()) / 2;
    const int32_t parts = static_cast<int32_t>(t.groupsPerBatch);
    if (group % parts != 0) {
        return;
    }
    const int32_t batch = group / parts;
    const int32_t co = static_cast<int32_t>(AscendC::GetSubBlockIdx()) * 2;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valuesBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> smallBuf;
    pipe.InitBuffer(rawBuf, 16 * 32 * 4);
    pipe.InitBuffer(valuesBuf, 7 * 64 * 4);
    pipe.InitBuffer(smallBuf, 64 * 4);
    auto raw = rawBuf.Get<float>();
    auto v = valuesBuf.Get<float>();
    auto ix = v[384].ReinterpretCast<int32_t>();
    AscendC::GlobalTensor<float> stats;
    AscendC::GlobalTensor<float> kg;
    AscendC::GlobalTensor<float> sg;
    const int32_t statsOff = static_cast<int32_t>(t.batch) * parts * static_cast<int32_t>(t.groupFloats);
    stats.SetGlobalBuffer((__gm__ float *)scratch + statsOff);
    kg.SetGlobalBuffer((__gm__ float *)keypoints);
    sg.SetGlobalBuffer((__gm__ float *)scores);
    AscendC::DataCopy(raw, stats[batch * parts * 32], parts * 32);
    AscendC::CreateVecIndex(ix, static_cast<int32_t>(0), REDUCE_LEN);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ix, ix, static_cast<int32_t>(128), REDUCE_LEN);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceVerticalChannels<REDUCE_LEN>(
        raw, ix.ReinterpretCast<uint32_t>(), v, smallBuf.Get<float>(), kg, sg, batch, co, parts, t);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunPackedCube(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    GM_ADDR scratch = AscendC::GetSysWorkSpacePtr() + t.workspaceOffset;
    if ASCEND_IS_AIC {
        FdkVerticalCubeStage<64, 1U, 2U, 9U> cubeOp;
        cubeOp.Run(feat, scratch, t);
    }
    if ASCEND_IS_AIV {
        FdkVerticalCube<64, 2, 6912, true, 1U, 2U, 9U> vecOp;
        vecOp.Run(weight, bias, scratch, t);
        AscendC::SyncAll();
        MergeVerticalCubeStats<8>(keypoints, scores, scratch, t);
    }
}

__aicore__ inline void RunHaloCube(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    GM_ADDR scratch = AscendC::GetSysWorkSpacePtr() + t.workspaceOffset;
    if ASCEND_IS_AIC {
        FdkVerticalCubeStage<64, 0U, 1U, 2U> cubeOp;
        cubeOp.Run(feat, scratch, t);
    }
    if ASCEND_IS_AIV {
        FdkVerticalCube<64, 2, 512, false, 0U, 1U, 2U> vecOp;
        vecOp.Run(weight, bias, scratch, t);
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        MergeVerticalCubeStats<64>(keypoints, scores, scratch, t);
    }
}
}  // namespace fdk

#endif  // FDK_VERTICAL_CUBE_H
