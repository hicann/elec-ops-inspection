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

#ifndef FDK_UNIFORM_OUTPUT_H
#define FDK_UNIFORM_OUTPUT_H

#include "fdk_fallback.h"

namespace fdk {
class FdkUniformOutput {
public:
    __aicore__ inline bool Run(
        GM_ADDR w,
        GM_ADDR b,
        GM_ADDR xy,
        GM_ADDR score,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        InitBuffers(w, b, xy, score);
        if (!AreAllWeightsZero()) {
            return false;
        }
        WriteUniformOutputs(t);
        return true;
    }

private:
    __aicore__ inline void InitBuffers(GM_ADDR w, GM_ADDR b, GM_ADDR xy, GM_ADDR score)
    {
        pipe_.InitBuffer(bitsBuffer_, 1152U * sizeof(uint32_t));
        pipe_.InitBuffer(numberBuffer_, 1152U * sizeof(float));
        pipe_.InitBuffer(reduceBuffer_, 1152U * sizeof(float));
        pipe_.InitBuffer(resultBuffer_, 128U * sizeof(float));
        bits_ = bitsBuffer_.Get<uint32_t>();
        number_ = numberBuffer_.Get<float>();
        scratch_ = reduceBuffer_.Get<float>();
        result_ = resultBuffer_.Get<float>();
        weight_.SetGlobalBuffer((__gm__ uint32_t *)w);
        bias_.SetGlobalBuffer((__gm__ float *)b);
        keypoints_.SetGlobalBuffer((__gm__ float *)xy);
        scores_.SetGlobalBuffer((__gm__ float *)score);
    }

    __aicore__ inline bool AreAllWeightsZero()
    {
        AscendC::DataCopy(bits_, weight_, 1152U);
        AscendC::DataCopyPad(
            result_[64], bias_, AscendC::DataCopyExtParams{1, 16, 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::ShiftLeft(bits_, bits_, static_cast<uint32_t>(1), 1152U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(bits_, bits_, static_cast<uint32_t>(1), 1152U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(number_, bits_.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, 1152U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax(result_[96], number_, scratch_, 1152U, false);
        AscendC::PipeBarrier<PIPE_ALL>();
        return result_.GetValue(96) == 0.0f;
    }

    __aicore__ inline void WriteUniformOutputs(const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        const float cx = static_cast<float>(static_cast<int32_t>(t.width)) * HALF_PIXEL_OFFSET * t.stride;
        const float cy = static_cast<float>(static_cast<int32_t>(t.height)) * HALF_PIXEL_OFFSET * t.stride;
        AscendC::Duplicate(result_, cx, 32U);
        uint64_t odd[2] = {0xAAAAAAAAULL, 0ULL};
        AscendC::Duplicate(result_, cy, odd, 1, 1, 8);
        auto index = bits_;
        AscendC::CreateVecIndex(index.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), 16U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(index, index, static_cast<uint32_t>(30), 16U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(index, index, static_cast<uint32_t>(28), 16U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(result_[32], result_[64], index, 0, 16U);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(keypoints_, result_, 32U);
        AscendC::DataCopy(scores_, result_[32], 16U);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bitsBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> numberBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultBuffer_;
    AscendC::LocalTensor<uint32_t> bits_;
    AscendC::LocalTensor<float> number_;
    AscendC::LocalTensor<float> scratch_;
    AscendC::LocalTensor<float> result_;
    AscendC::GlobalTensor<uint32_t> weight_;
    AscendC::GlobalTensor<float> bias_;
    AscendC::GlobalTensor<float> keypoints_;
    AscendC::GlobalTensor<float> scores_;
};

__aicore__ inline void RunUniformOutput(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    {
        FdkUniformOutput probe;
        if (probe.Run(weight, bias, keypoints, scores, t)) {
            return;
        }
    }
    const uint32_t totalPlanes = t.batch * 4U;
    for (uint32_t plane = 0U; plane < totalPlanes; ++plane) {
        RunFallbackPlane(feat, weight, bias, keypoints, scores, t, static_cast<int32_t>(plane));
    }
}
}  // namespace fdk

#endif  // FDK_UNIFORM_OUTPUT_H
