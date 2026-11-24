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

#ifndef FDK_COMMON_H
#define FDK_COMMON_H

#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"

namespace matmul {
__aicore__ inline void clearWorkspace(GM_ADDR)
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetLoadDataBoundary(static_cast<uint64_t>(0));
    AscendC::SetLoadDataPaddingValue(static_cast<uint64_t>(0));
}
}  // namespace matmul

namespace fdk {
constexpr float NEG_MAX_F32 = -3.402823466e38f;
constexpr float POS_MAX_F32 = 3.402823466e38f;
constexpr float HALF_PIXEL_OFFSET = 0.5f;

__aicore__ inline float SafeDiv(float num, float den)
{
    const float safeDen = (den == 0.0f) ? 1.0f : den;
    return num / safeDen;
}

template <typename T>
__aicore__ inline void FillSequence(AscendC::LocalTensor<T> dst, int32_t count)
{
    for (int32_t i = 0; (i < 8) && (i < count); ++i) {
        dst.SetValue(i, static_cast<T>(i));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t span = 8; span < count; span *= 2) {
        const int32_t rem = count - span;
        const int32_t len = (rem < span) ? rem : span;
        AscendC::Adds(dst[span], dst, static_cast<T>(span), len);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void StoreKeypointScore(
    AscendC::LocalTensor<float> small,
    AscendC::GlobalTensor<float> kg,
    AscendC::GlobalTensor<float> sg,
    int32_t plane,
    float x,
    float y,
    float score)
{
    small.SetValue(32, x);
    small.SetValue(33, y);
    small.SetValue(48, score);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(kg[plane * 2], small[32], AscendC::DataCopyExtParams{1, 8, 0, 0, 0});
    AscendC::DataCopyPad(sg[plane], small[48], AscendC::DataCopyExtParams{1, 4, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void PackProjectionWeights(
    AscendC::LocalTensor<float> raw,
    AscendC::LocalTensor<float> temp,
    AscendC::LocalTensor<uint32_t> ix,
    AscendC::GlobalTensor<float> wg,
    AscendC::GlobalTensor<float> packed,
    int32_t sub)
{
    AscendC::DataCopy(raw, wg[sub * 576], 576);
    auto ixI32 = ix.ReinterpretCast<int32_t>();
    FillSequence(ixI32, 32);
    AscendC::Muls(ixI32, ixI32, static_cast<int32_t>(36), 32);
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t k = 0; k < 2; ++k) {
        for (int32_t dx = 0; dx < 3; ++dx) {
            for (int32_t dy = 0; dy < 3; ++dy) {
                const int32_t dst = (k * 3 + dx) * 96 + dy * 32;
                if (dst != 0) {
                    const int32_t delta = (k * 288 + dy * 3 + dx) * 4;
                    AscendC::Adds(ix[dst].ReinterpretCast<int32_t>(), ixI32, delta, 32);
                }
            }
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(temp, raw, ix, 0, 576);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopy(packed[sub * 576], temp, 576);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline float ExpShiftInPlace(
    AscendC::LocalTensor<float> values,
    AscendC::LocalTensor<float> reduction,
    AscendC::LocalTensor<float> small,
    int32_t count,
    float invTau)
{
    AscendC::ReduceMax(small, values, reduction, count, false);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    const float peak = small.GetValue(0);
    AscendC::Adds(values, values, -peak, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(values, values, invTau, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(values, values, count);
    AscendC::PipeBarrier<PIPE_V>();
    return peak;
}

struct StatMergeResult {
    float peak;
    float den;
    float sumX;
    float sumY;
};

__aicore__ inline StatMergeResult ReduceMergedStats(
    AscendC::LocalTensor<float> maximum,
    AscendC::LocalTensor<float> mass,
    AscendC::LocalTensor<float> xmoment,
    AscendC::LocalTensor<float> ymoment,
    AscendC::LocalTensor<float> temp,
    AscendC::LocalTensor<float> reduction,
    AscendC::LocalTensor<float> small,
    int32_t count,
    float invTau)
{
    const float peak = ExpShiftInPlace(maximum, reduction, small, count, invTau);
    AscendC::Mul(temp, mass, maximum, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(small[8], temp, reduction, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(temp, xmoment, maximum, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(small[16], temp, reduction, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(temp, ymoment, maximum, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(small[24], temp, reduction, count);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    StatMergeResult res{};
    res.peak = peak;
    res.den = small.GetValue(8);
    res.sumX = small.GetValue(16);
    res.sumY = small.GetValue(24);
    return res;
}

__aicore__ inline void GatherStatLanes(
    AscendC::LocalTensor<float> raw,
    AscendC::LocalTensor<uint32_t> gatherIdx,
    AscendC::LocalTensor<float> maximum,
    AscendC::LocalTensor<float> mass,
    AscendC::LocalTensor<float> xmoment,
    AscendC::LocalTensor<float> ymoment,
    int32_t channel,
    int32_t parts,
    int32_t padCount)
{
    if (padCount > parts) {
        AscendC::Duplicate(maximum, NEG_MAX_F32, padCount);
        AscendC::Duplicate(mass, 0.0f, padCount);
        AscendC::Duplicate(xmoment, 0.0f, padCount);
        AscendC::Duplicate(ymoment, 0.0f, padCount);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const uint32_t base = static_cast<uint32_t>(channel * 32);
    AscendC::Gather(maximum, raw, gatherIdx, base, parts);
    AscendC::Gather(mass, raw, gatherIdx, base + 4U, parts);
    AscendC::Gather(xmoment, raw, gatherIdx, base + 8U, parts);
    AscendC::Gather(ymoment, raw, gatherIdx, base + 12U, parts);
    AscendC::PipeBarrier<PIPE_V>();
}
}  // namespace fdk

#endif  // FDK_COMMON_H
