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

#ifndef FDK_FALLBACK_H
#define FDK_FALLBACK_H

#include "fdk_common.h"

namespace fdk {
// Generic FP32 vector path: one AIV owns one complete (batch, keypoint) heatmap.
class FdkFallback {
public:
    __aicore__ inline void Init(
        GM_ADDR feat,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR keypoints,
        GM_ADDR scores,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &td,
        int32_t explicitPlane = -1)
    {
        h_ = td.height;
        w_ = td.width;
        n_ = h_ * w_;
        plane_ = (explicitPlane >= 0) ? static_cast<uint32_t>(explicitPlane) : AscendC::GetBlockIdx();
        batch_ = plane_ / 4U;
        key_ = plane_ % 4U;
        invTau_ = td.invTau;
        stride_ = td.stride;
        feat_.SetGlobalBuffer((__gm__ float *)feat);
        weight_.SetGlobalBuffer((__gm__ float *)weight);
        bias_.SetGlobalBuffer((__gm__ float *)bias);
        keypoints_.SetGlobalBuffer((__gm__ float *)keypoints);
        scores_.SetGlobalBuffer((__gm__ float *)scores);
        pipe_.InitBuffer(input_, 2, (n_ + 2U * w_) * sizeof(float));
        pipe_.InitBuffer(acc_, n_ * sizeof(float));
        pipe_.InitBuffer(temp_, n_ * sizeof(float));
        pipe_.InitBuffer(scratch_, n_ * sizeof(float));
        pipe_.InitBuffer(weights_, 288U * sizeof(float));
        pipe_.InitBuffer(small_, 512U * sizeof(float));
    }

    __aicore__ inline void Run()
    {
        auto weights = weights_.Get<float>();
        auto small = small_.Get<float>();
        auto temp = temp_.Get<float>();
        auto scratch = scratch_.Get<float>();
        AscendC::DataCopy(weights, weight_[key_ * 288U], 288U);
        AscendC::DataCopyPad(
            small[480], bias_[key_], AscendC::DataCopyExtParams{1, 4, 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        AscendC::PipeBarrier<PIPE_ALL>();
        const float biasValue = small.GetValue(480);
        AscendC::Abs(temp, weights, 288U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax(small, temp, scratch, 288U, false);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (small.GetValue(0) == 0.0f) {
            const float cx = static_cast<float>(static_cast<int32_t>(w_)) * HALF_PIXEL_OFFSET * stride_;
            const float cy = static_cast<float>(static_cast<int32_t>(h_)) * HALF_PIXEL_OFFSET * stride_;
            Store(cx, cy, biasValue);
            return;
        }
        Convolve();
        DecodeAndStore(biasValue);
    }

private:
    __aicore__ inline void Convolve()
    {
        auto acc = acc_.Get<float>();
        auto weights = weights_.Get<float>();
        AscendC::Duplicate(acc, 0.0f, n_);
        AscendC::PipeBarrier<PIPE_V>();
        LoadHorizontal(0U);
        for (uint32_t item = 0U; item < 96U; ++item) {
            if (item + 1U < 96U) {
                LoadHorizontal(item + 1U);
            }
            auto in = input_.DeQue<float>();
            const uint32_t channel = item / 3U;
            const uint32_t dx = item % 3U;
            const uint32_t coeff = channel * 9U + dx;
            const float a0 = weights.GetValue(coeff);
            const float a1 = weights.GetValue(coeff + 3U);
            const float a2 = weights.GetValue(coeff + 6U);
            AscendC::Axpy(acc, in, a0, n_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Axpy(acc, in[w_], a1, n_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Axpy(acc, in[2U * w_], a2, n_);
            input_.FreeTensor(in);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecodeAndStore(float biasValue)
    {
        auto acc = acc_.Get<float>();
        auto temp = temp_.Get<float>();
        auto scratch = scratch_.Get<float>();
        auto small = small_.Get<float>();
        AscendC::ReduceMax(small, acc, scratch, n_, false);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float rawMax = small.GetValue(0);
        AscendC::Adds(acc, acc, -rawMax, n_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(acc, acc, invTau_, n_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(acc, acc, n_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(small[128], acc, static_cast<uint64_t>(64), h_, 1, 1, w_ / 8U);
        if (w_ > 64U) {
            AscendC::WholeReduceSum(small[256], acc[64], static_cast<uint64_t>(w_ - 64U), h_, 1, 1, w_ / 8U);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(small[128], small[128], small[256], h_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small, small[128], scratch, h_);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float denominator = small.GetValue(0);
        ComputeExpectedCoords(acc, temp, scratch, small, denominator, rawMax + biasValue);
    }

    __aicore__ inline void ComputeExpectedCoords(
        AscendC::LocalTensor<float> acc,
        AscendC::LocalTensor<float> temp,
        AscendC::LocalTensor<float> scratch,
        AscendC::LocalTensor<float> small,
        float denominator,
        float score)
    {
        for (uint32_t x = 0U; x < w_; ++x) {
            small.SetValue(256U + x, static_cast<float>(static_cast<int32_t>(x)));
        }
        for (uint32_t y = 0U; y < h_; ++y) {
            small.SetValue(384U + y, static_cast<float>(static_cast<int32_t>(y)));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        const AscendC::BinaryRepeatParams rep{
            1, 1, 1, static_cast<uint8_t>(w_ / 8U), static_cast<uint8_t>(w_ / 8U), 0};
        AscendC::Mul(temp, acc, small[256], static_cast<uint64_t>(64), h_, rep);
        if (w_ > 64U) {
            AscendC::Mul(temp[64], acc[64], small[320], static_cast<uint64_t>(w_ - 64U), h_, rep);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small, temp, scratch, n_);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float xCoord = (SafeDiv(small.GetValue(0), denominator) + HALF_PIXEL_OFFSET) * stride_;
        AscendC::Mul(small[128], small[128], small[384], h_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(small, small[128], scratch, h_);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float yCoord = (SafeDiv(small.GetValue(0), denominator) + HALF_PIXEL_OFFSET) * stride_;
        Store(xCoord, yCoord, score);
    }

    __aicore__ inline void LoadHorizontal(uint32_t item)
    {
        auto in = input_.AllocTensor<float>();
        const uint32_t channel = item / 3U;
        const uint32_t dx = item % 3U;
        AscendC::Duplicate(in, 0.0f, w_);
        AscendC::Duplicate(in[n_ + w_], 0.0f, w_);
        const uint32_t columns = w_ - static_cast<uint32_t>(dx != 1U);
        const uint32_t offset = (batch_ * 32U + channel) * n_ + static_cast<uint32_t>(dx == 2U);
        const AscendC::DataCopyExtParams ext{
            static_cast<uint16_t>(h_), columns * 4U, (w_ - columns) * 4U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> pad{
            true, static_cast<uint8_t>(dx == 0U), static_cast<uint8_t>(dx == 2U), 0.0f};
        AscendC::DataCopyPad(in[w_], feat_[offset], ext, pad);
        input_.EnQue(in);
    }

    __aicore__ inline void Store(float x, float y, float score)
    {
        auto small = small_.Get<float>();
        small.SetValue(0, x);
        small.SetValue(1, y);
        small.SetValue(8, score);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(keypoints_[plane_ * 2U], small, AscendC::DataCopyExtParams{1, 8, 0, 0, 0});
        AscendC::DataCopyPad(scores_[plane_], small[8], AscendC::DataCopyExtParams{1, 4, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> input_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> acc_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> temp_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scratch_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> weights_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> small_;
    AscendC::GlobalTensor<float> feat_;
    AscendC::GlobalTensor<float> weight_;
    AscendC::GlobalTensor<float> bias_;
    AscendC::GlobalTensor<float> keypoints_;
    AscendC::GlobalTensor<float> scores_;
    uint32_t h_ = 0U;
    uint32_t w_ = 0U;
    uint32_t n_ = 0U;
    uint32_t plane_ = 0U;
    uint32_t batch_ = 0U;
    uint32_t key_ = 0U;
    float invTau_ = 1.0f;
    float stride_ = 4.0f;
};

__aicore__ inline void RunFallback(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &td)
{
    FdkFallback op;
    op.Init(feat, weight, bias, keypoints, scores, td);
    op.Run();
}

__aicore__ inline void RunFallbackPlane(
    GM_ADDR feat,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR keypoints,
    GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &td,
    int32_t plane)
{
    FdkFallback op;
    op.Init(feat, weight, bias, keypoints, scores, td, plane);
    op.Run();
}
}  // namespace fdk

#endif  // FDK_FALLBACK_H
