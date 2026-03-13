// Copyright 2026 Electrical Engineering SIG - CANN Community
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace NsOptimizedTransducer {

using namespace AscendC;

constexpr int BUFFER_NUM = 2;          // Double buffer for TPipe
constexpr uint32_t ALIGN_ELEMENTS = 8; // 32字节对齐，float类型需要8个元素对齐

#pragma region Class_Definition
class OptimizedTransducer {
public:
    __aicore__ inline OptimizedTransducer(){};

    __aicore__ inline void Init(
        GM_ADDR logits, GM_ADDR targets, GM_ADDR logitLengths, GM_ADDR targetLengths, GM_ADDR loss, GM_ADDR grad,
        GM_ADDR workspace, const OptimizedTransducerTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GetCurrentTU(int32_t sampleIdx, int32_t& T, int32_t& U);
    template <HardEvent event>
    __aicore__ inline void Sync(int8_t id = 0);
    __aicore__ inline float ScalarLog(float x);
    __aicore__ inline float ScalarExp(float x);
    __aicore__ inline float ScalarAbs(float x);
    __aicore__ inline void ComputeLogSoftmax(int32_t sampleIdx);
    __aicore__ inline void ComputeAlpha(int32_t sampleIdx);
    __aicore__ inline void ComputeBeta(int32_t sampleIdx);
    __aicore__ inline void ComputeGrad(int32_t sampleIdx);
    __aicore__ inline float LogAdd(float a, float b);
    __aicore__ inline void ComputeLogSoftmaxSingleLargeVector(int64_t vectorIdx, uint32_t tileLen);
    __aicore__ inline void ComputeLogSoftmaxForBatchVector(
        int64_t startVectorIdx, int32_t vectorCount, uint32_t tileLen);
    __aicore__ inline void ComputeGradForBatchVector(
        int64_t startVectorIdx, int32_t vectorCount, uint32_t tileLen, int32_t sampleIdx);
    __aicore__ inline void ComputeGradSingleLargeVector(
        int64_t vectorIdx, uint32_t tileLen, float logGamma, float gradBlankPosterior, float gradLabelPosterior,
        int32_t targetLabel);
    __aicore__ inline int64_t GetPosOffsetLocal(int32_t t, int32_t u);
    __aicore__ inline GlobalTensor<float>& GetLogProbGm();
    __aicore__ inline int32_t GetTargetLabel(int32_t sample, int32_t targetIdx);
    __aicore__ inline void LoadProbRow(
        int32_t t, uint32_t uSizeAligned, LocalTensor<int32_t>& targetsLocal, LocalTensor<float>& probIn);
    __aicore__ inline void StoreBetaRow(int32_t t, uint32_t uSizeAligned, LocalTensor<float>& betaRowSrc);

private:
    GlobalTensor<float> logitsGm;
    GlobalTensor<int32_t> targetsGm;
    GlobalTensor<int32_t> logitLengthsGm;
    GlobalTensor<int32_t> targetLengthsGm;
    GlobalTensor<float> lossGm;
    GlobalTensor<float> gradGm;
    GlobalTensor<float> alphaGm;
    GlobalTensor<float> betaGm;

    int32_t blankIdx_ = 0;
    float clamp_ = -1.0f;
    bool fusedLogSoftmax_ = true;
    int32_t vocabSize_ = 0;
    uint32_t alignedVocabSize_ = 0; // 对齐后的 vocab size

    int32_t sampleStart_ = 0;
    int32_t sampleEnd_ = 0;
    int32_t maxTargetLength_ = 0;

    int32_t T_ = 0;
    int32_t U_ = 0;

    int64_t currentSampleOffset_ = 0;
    int64_t ubSize_ = 0;
    uint32_t scalarSize_ = 128;

    LocalTensor<float> scalarLocal_;

    float cachedLoss_ = 0.0f;

    // TPipe for LogSoftmax processing
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<QuePosition::VECCALC> workBuf_;

    uint32_t logSoftmaxTileLen_ = 0;
};
#pragma endregion

#pragma region Initialization
__aicore__ inline void OptimizedTransducer::Init(
    GM_ADDR logits, GM_ADDR targets, GM_ADDR logitLengths, GM_ADDR targetLengths, GM_ADDR loss, GM_ADDR grad,
    GM_ADDR workspace, const OptimizedTransducerTilingData* tilingData)
{
    vocabSize_ = static_cast<int32_t>(tilingData->vocabSize);
    // 向上对齐到 8 的倍数（32字节/4字节=8）
    alignedVocabSize_ = ((vocabSize_ + ALIGN_ELEMENTS - 1) / ALIGN_ELEMENTS) * ALIGN_ELEMENTS;

    int32_t blockIdx = GetBlockIdx();
    int32_t blockNum = tilingData->usedCoreNum;
    int64_t blank = tilingData->blank;

    ubSize_ = tilingData->ubSize;
    blankIdx_ = (blank < 0) ? (vocabSize_ - 1) : static_cast<int32_t>(blank);
    clamp_ = tilingData->clamp;
    fusedLogSoftmax_ = tilingData->fusedLogSoftmax;
    int32_t batchSize = static_cast<int32_t>(tilingData->batchSize);
    maxTargetLength_ = static_cast<int32_t>(tilingData->maxTargetLength);

    int32_t samplesPerBlock = batchSize / blockNum;
    int32_t sampleRemainder = batchSize % blockNum;
    sampleStart_ = blockIdx * samplesPerBlock + (blockIdx < sampleRemainder ? blockIdx : sampleRemainder);
    sampleEnd_ = sampleStart_ + samplesPerBlock + (blockIdx < sampleRemainder ? 1 : 0);

    // 初始化 Global Tensor
    int64_t totalElements = tilingData->totalPositions * vocabSize_;
    logitsGm.SetGlobalBuffer((__gm__ float*)logits, totalElements);
    gradGm.SetGlobalBuffer((__gm__ float*)grad, totalElements);
    lossGm.SetGlobalBuffer((__gm__ float*)loss, batchSize);
    targetsGm.SetGlobalBuffer((__gm__ int32_t*)targets, batchSize * maxTargetLength_);
    logitLengthsGm.SetGlobalBuffer((__gm__ int32_t*)logitLengths, batchSize);
    targetLengthsGm.SetGlobalBuffer((__gm__ int32_t*)targetLengths, batchSize);

    int64_t totalPositions = tilingData->totalPositions;
    alphaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), totalPositions);
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + totalPositions, totalPositions);

    // 计算 currentSampleOffset_
    for (int32_t s = 0; s < sampleStart_; ++s) {
        int32_t T, U;
        GetCurrentTU(s, T, U);
        currentSampleOffset_ += static_cast<int64_t>(T) * U;
    }
}
#pragma endregion

#pragma region LogSoftmax
__aicore__ inline void OptimizedTransducer::ComputeLogSoftmaxForBatchVector(
    int64_t startVectorIdx, int32_t vectorCount, uint32_t tileLen)
{
    int64_t gmOffset = (currentSampleOffset_ + startVectorIdx) * vocabSize_;

    // Stage 1: Copy In
    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
    for (int32_t v = 0; v < vectorCount; ++v) {
        int64_t srcOffset = gmOffset + v * vocabSize_;
        uint32_t dstOffset = v * alignedVocabSize_;
        DataCopy(inLocal[dstOffset], logitsGm[srcOffset], alignedVocabSize_);
    }
    inQueue_.EnQue(inLocal);

    // Stage 2: Compute
    inLocal = inQueue_.DeQue<float>();
    LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();

    for (int32_t v = 0; v < vectorCount; ++v) {
        uint32_t vOffset = v * alignedVocabSize_;
        LocalTensor<float> vInput = inLocal[vOffset];
        LocalTensor<float> vOutput = outLocal[vOffset];

        // 1. 计算最大值
        ReduceMax(scalarLocal_, vInput, workLocal, vocabSize_);
        pipe_barrier(PIPE_V);
        float vMax = scalarLocal_.GetValue(0);

        // 2. 减去最大值并计算exp
        Adds(vOutput, vInput, -vMax, vocabSize_);
        Exp(vOutput, vOutput, vocabSize_);

        // 3. 计算sum
        ReduceSum(scalarLocal_, vOutput, workLocal, vocabSize_);
        pipe_barrier(PIPE_V);
        float vSum = scalarLocal_.GetValue(0);

        // 4. 计算最终的log softmax: x - max - log(sum)
        float finalBias = -vMax - ScalarLog(vSum);
        Adds(vOutput, vInput, finalBias, vocabSize_);
    }

    outQueue_.EnQue(outLocal);
    inQueue_.FreeTensor(inLocal);

    // Stage 3: Copy Out
    outLocal = outQueue_.DeQue<float>();
    for (int32_t v = 0; v < vectorCount; ++v) {
        uint32_t srcOffset = v * alignedVocabSize_;
        int64_t dstOffset = gmOffset + v * vocabSize_;
        uint32_t copySize = vocabSize_ * sizeof(float);

        DataCopyExtParams copyOutParams{1, copySize, 0, 0, 0};
        DataCopyPad(gradGm[dstOffset], outLocal[srcOffset], copyOutParams);
    }
    outQueue_.FreeTensor(outLocal);
}

__aicore__ inline void OptimizedTransducer::ComputeLogSoftmaxSingleLargeVector(int64_t vectorIdx, uint32_t tileLen)
{
    int64_t gmOffset = (currentSampleOffset_ + vectorIdx) * vocabSize_;
    uint32_t loopCount = (vocabSize_ + tileLen - 1) / tileLen;

    LocalTensor<float> workLocal = workBuf_.Get<float>();

    float globalMax = -1.0e20f;
    float globalSum = 0.0f;

    // Pass 1: 计算全局最大值和sum
    for (uint32_t i = 0; i < loopCount; ++i) {
        int32_t curOffset = i * tileLen;
        int32_t curLen = (i == loopCount - 1) ? (vocabSize_ - curOffset) : tileLen;
        uint32_t copySize = curLen * sizeof(float);

        // Copy In
        LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
        DataCopyExtParams copyParams{1, copySize, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, -1.0e20f};
        DataCopyPad(inLocal, logitsGm[gmOffset + curOffset], copyParams, padParams);
        inQueue_.EnQue(inLocal);

        // Compute max and partial sum
        inLocal = inQueue_.DeQue<float>();
        LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();

        ReduceMax(scalarLocal_, inLocal, workLocal, curLen);
        pipe_barrier(PIPE_V);
        float localMax = scalarLocal_.GetValue(0);

        Adds(outLocal, inLocal, -localMax, curLen);
        Exp(outLocal, outLocal, curLen);
        ReduceSum(scalarLocal_, outLocal, workLocal, curLen);
        pipe_barrier(PIPE_V);
        float localSum = scalarLocal_.GetValue(0);

        // 更新全局最大值和sum（数值稳定的方式）
        if (localMax > globalMax) {
            float correction = ScalarExp(globalMax - localMax);
            globalSum = globalSum * correction + localSum;
            globalMax = localMax;
        } else {
            float correction = ScalarExp(localMax - globalMax);
            globalSum = globalSum + localSum * correction;
        }

        outQueue_.EnQue(outLocal);
        inQueue_.FreeTensor(inLocal);

        // 释放outLocal（这一轮的结果不需要写回，只是用来计算sum）
        outLocal = outQueue_.DeQue<float>();
        outQueue_.FreeTensor(outLocal);
    }

    float finalBias = -globalMax - ScalarLog(globalSum);

    // Pass 2: 应用最终的bias并写回
    for (uint32_t i = 0; i < loopCount; ++i) {
        int32_t curOffset = i * tileLen;
        int32_t curLen = (i == loopCount - 1) ? (vocabSize_ - curOffset) : tileLen;
        uint32_t copySize = curLen * sizeof(float);

        // Copy In
        LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
        DataCopyExtParams copyParams{1, copySize, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, -1.0e20f};
        DataCopyPad(inLocal, logitsGm[gmOffset + curOffset], copyParams, padParams);
        inQueue_.EnQue(inLocal);

        // Compute final result
        inLocal = inQueue_.DeQue<float>();
        LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();

        Adds(outLocal, inLocal, finalBias, curLen);

        outQueue_.EnQue(outLocal);
        inQueue_.FreeTensor(inLocal);

        // Copy Out
        outLocal = outQueue_.DeQue<float>();
        DataCopyPad(gradGm[gmOffset + curOffset], outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }
}

__aicore__ inline void OptimizedTransducer::ComputeLogSoftmax(int32_t sampleIdx)
{
    if (!fusedLogSoftmax_) {
        return;
    }

    int64_t totalPos = static_cast<int64_t>(T_) * U_;
    uint32_t bufferSize = (ubSize_ - scalarSize_) / 5 / 32 * 32;
    uint32_t tileLen = bufferSize / sizeof(float);

    // 初始化TPipe
    pipe_.InitBuffer(inQueue_, BUFFER_NUM, bufferSize);   // 2
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, bufferSize);  // 2
    pipe_.InitBuffer(workBuf_, bufferSize + scalarSize_); // 1 + scalarLocal_

    // 分配 scalarLocal_
    scalarLocal_ = workBuf_.GetWithOffset<float>(scalarSize_ / sizeof(float), bufferSize);

    if (alignedVocabSize_ < tileLen) {
        uint32_t vectorsPerBatch = tileLen / alignedVocabSize_;
        for (int64_t i = 0; i < totalPos; i += vectorsPerBatch) {
            uint32_t currentBatchSize = (i + vectorsPerBatch > totalPos) ? (totalPos - i) : vectorsPerBatch;
            ComputeLogSoftmaxForBatchVector(i, currentBatchSize, tileLen);
        }
    } else {
        for (int64_t i = 0; i < totalPos; ++i) {
            ComputeLogSoftmaxSingleLargeVector(i, tileLen);
        }
    }
    pipe_.Reset();
}

#pragma endregion

#pragma region Helpers
__aicore__ inline void OptimizedTransducer::GetCurrentTU(int32_t sampleIdx, int32_t& T, int32_t& U)
{
    T = logitLengthsGm.GetValue(sampleIdx);
    U = targetLengthsGm.GetValue(sampleIdx) + 1;
    Sync<HardEvent::MTE2_S>();
}

template <HardEvent event>
__aicore__ inline void OptimizedTransducer::Sync(int8_t id)
{
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

__aicore__ inline int64_t OptimizedTransducer::GetPosOffsetLocal(int32_t t, int32_t u)
{
    return (currentSampleOffset_ + t * U_ + u) * vocabSize_;
}

__aicore__ inline GlobalTensor<float>& OptimizedTransducer::GetLogProbGm()
{
    return fusedLogSoftmax_ ? gradGm : logitsGm;
}

__aicore__ inline int32_t OptimizedTransducer::GetTargetLabel(int32_t sample, int32_t targetIdx)
{
    return targetsGm.GetValue(sample * maxTargetLength_ + targetIdx);
    Sync<HardEvent::MTE2_S>();
}

__aicore__ inline float OptimizedTransducer::ScalarAbs(float x)
{
    return (x >= 0.0f) ? x : -x;
}

__aicore__ inline float OptimizedTransducer::ScalarLog(float x)
{
    scalarLocal_.SetValue(0, x);
    AscendC::Ln(scalarLocal_, scalarLocal_, 1);
    return scalarLocal_.GetValue(0);
}

__aicore__ inline float OptimizedTransducer::ScalarExp(float x)
{
    scalarLocal_.SetValue(0, x);
    AscendC::Exp(scalarLocal_, scalarLocal_, 1);
    return scalarLocal_.GetValue(0);
}

__aicore__ inline float OptimizedTransducer::LogAdd(float a, float b)
{
    // LogAdd(a, b) = log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    float maxVal = (a > b) ? a : b;
    float diff = ScalarAbs(a - b);
    if (diff > 50.0f) {
        return maxVal;
    }

    float expNegDiff = ScalarExp(-diff);

    // if (diff > 8.0f) {
    //     return maxVal + expNegDiff;
    // }

    float logTerm = ScalarLog(1.0f + expNegDiff);

    return maxVal + logTerm;
}
__aicore__ inline void OptimizedTransducer::LoadProbRow(
    int32_t t, uint32_t uSizeAligned, LocalTensor<int32_t>& targetsLocal, LocalTensor<float>& probIn)
{
    GlobalTensor<float>& logProbGm = GetLogProbGm();

    // 加载 logPBlank
    LocalTensor<float> logPBlankPart = probIn;
    for (int32_t u = 0; u < U_; ++u) {
        logPBlankPart.SetValue(u, logProbGm.GetValue(GetPosOffsetLocal(t, u) + blankIdx_));
    }

    // 加载 logPLabel
    LocalTensor<float> logPLabelPart = probIn[uSizeAligned];
    for (int32_t u = 0; u < U_ - 1; ++u) {
        int32_t label = targetsLocal.GetValue(u);
        logPLabelPart.SetValue(u, logProbGm.GetValue(GetPosOffsetLocal(t, u) + label));
    }
}

__aicore__ inline void OptimizedTransducer::StoreBetaRow(
    int32_t t, uint32_t uSizeAligned, LocalTensor<float>& betaRowSrc)
{
    LocalTensor<float> betaOut = outQueue_.AllocTensor<float>();
    DataCopy(betaOut, betaRowSrc, uSizeAligned);
    outQueue_.EnQue(betaOut);
    betaOut = outQueue_.DeQue<float>();
    DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(U_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(betaGm[currentSampleOffset_ + t * U_], betaOut, outCopyParams);
    outQueue_.FreeTensor(betaOut);
}
#pragma endregion

#pragma region Alpha
__aicore__ inline void OptimizedTransducer::ComputeAlpha(int32_t sampleIdx)
{
    uint32_t uSizeAligned = (U_ + 7) / 8 * 8;
    uint32_t rowBytes = uSizeAligned * sizeof(float);

    // TPipe 空间规划:
    // - inQueue_: 概率行输入 (logPBlank + logPLabel = 2 * rowBytes)
    // - outQueue_: alpha 输出
    // - workBuf_: targets + 2 rows alpha + logPBlankPrev + scalarLocal_
    uint32_t probRowBytes = 2 * rowBytes;
    uint32_t workBytes = 4 * rowBytes + scalarSize_;

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, probRowBytes);
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, rowBytes);
    pipe_.InitBuffer(workBuf_, workBytes);

    scalarLocal_ = workBuf_.GetWithOffset<float>(scalarSize_ / sizeof(float), 4 * rowBytes);

    LocalTensor<int32_t> targetsLocal = workBuf_.Get<int32_t>(rowBytes / sizeof(int32_t));
    LocalTensor<float> alphaRow[2];
    alphaRow[0] = workBuf_.GetWithOffset<float>(uSizeAligned, rowBytes);
    alphaRow[1] = workBuf_.GetWithOffset<float>(uSizeAligned, 2 * rowBytes);
    LocalTensor<float> logPBlankPrev = workBuf_.GetWithOffset<float>(uSizeAligned, 3 * rowBytes);

    // 1. 加载 targets
    DataCopyExtParams targetCopyParams{1, static_cast<uint32_t>((U_ - 1) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> targetPadParams{false, 0, 0, 0};
    DataCopyPad(targetsLocal, targetsGm[sampleIdx * maxTargetLength_], targetCopyParams, targetPadParams);
    Sync<HardEvent::MTE2_S>();

    // 2. 加载 t=0 行概率
    LocalTensor<float> probIn = inQueue_.AllocTensor<float>();
    LoadProbRow(0, uSizeAligned, targetsLocal, probIn);
    inQueue_.EnQue(probIn);

    // 3. 计算边界行 (t = 0)
    // alpha(0, 0) = 0, alpha(0, u) = alpha(0, u-1) + logPLabel(0, u-1)
    probIn = inQueue_.DeQue<float>();
    LocalTensor<float> logPBlank = probIn;
    LocalTensor<float> logPLabel = probIn[uSizeAligned];

    int32_t currIdx = 0;
    alphaRow[currIdx].SetValue(0, 0.0f);
    for (int32_t u = 1; u < U_; ++u) {
        alphaRow[currIdx].SetValue(u, alphaRow[currIdx].GetValue(u - 1) + logPLabel.GetValue(u - 1));
    }

    // 保存 t=0 的 logPBlank 供 t=1 使用
    DataCopy(logPBlankPrev, logPBlank, uSizeAligned);
    inQueue_.FreeTensor(probIn);

    // 输出第一行
    LocalTensor<float> alphaOut = outQueue_.AllocTensor<float>();
    DataCopy(alphaOut, alphaRow[currIdx], uSizeAligned);
    outQueue_.EnQue(alphaOut);
    alphaOut = outQueue_.DeQue<float>();
    DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(U_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(alphaGm[currentSampleOffset_], alphaOut, outCopyParams);
    outQueue_.FreeTensor(alphaOut);

    // 4. 主迭代 (t = 1 to T-1)
    // alpha(t, u) = LogAdd(alpha(t-1, u) + logPBlank(t-1, u), alpha(t, u-1) + logPLabel(t, u-1))
    for (int32_t t = 1; t < T_; ++t) {
        int32_t nextIdx = 1 - currIdx;

        // 加载当前 t 行的概率
        LocalTensor<float> probIn = inQueue_.AllocTensor<float>();
        LoadProbRow(t, uSizeAligned, targetsLocal, probIn);
        inQueue_.EnQue(probIn);
        probIn = inQueue_.DeQue<float>();
        LocalTensor<float> logPBlankCurr = probIn;
        LocalTensor<float> logPLabelCurr = probIn[uSizeAligned];

        // 边界: u = 0, 只能从 t-1 方向转移
        alphaRow[nextIdx].SetValue(0, alphaRow[currIdx].GetValue(0) + logPBlankPrev.GetValue(0));

        // 其余位置: 两个方向的 LogAdd
        for (int32_t u = 1; u < U_; ++u) {
            float alphaFromT = alphaRow[currIdx].GetValue(u) + logPBlankPrev.GetValue(u);
            float alphaFromU = alphaRow[nextIdx].GetValue(u - 1) + logPLabelCurr.GetValue(u - 1);
            alphaRow[nextIdx].SetValue(u, LogAdd(alphaFromT, alphaFromU));
        }

        // 更新 logPBlankPrev 为当前行，供下一轮使用
        DataCopy(logPBlankPrev, logPBlankCurr, uSizeAligned);
        pipe_barrier(PIPE_V);
        inQueue_.FreeTensor(probIn);

        // 输出当前行
        alphaOut = outQueue_.AllocTensor<float>();
        DataCopy(alphaOut, alphaRow[nextIdx], uSizeAligned);
        outQueue_.EnQue(alphaOut);
        alphaOut = outQueue_.DeQue<float>();
        DataCopyPad(alphaGm[currentSampleOffset_ + t * U_], alphaOut, outCopyParams);
        outQueue_.FreeTensor(alphaOut);

        currIdx = nextIdx;
    }

    /** 
    // 对比 alpha 和 beta 计算得到的 loss
    // 从 alpha 计算: loss = -(alpha(T-1, U-1) + logPBlank(T-1, U-1))
    GlobalTensor<float>& logProbGm = GetLogProbGm();
    int64_t lastPosIdx = currentSampleOffset_ + (T_ - 1) * U_ + (U_ - 1);
    float alphaLast = alphaGm.GetValue(lastPosIdx);
    float logPBlankLast = logProbGm.GetValue(lastPosIdx * vocabSize_ + blankIdx_);
    Sync<HardEvent::MTE2_S>();
    
    float lossFromAlpha = -(alphaLast + logPBlankLast);
    float lossFromBeta = cachedLoss_;
    float lossError = ScalarAbs(lossFromAlpha - lossFromBeta);
    float lossRelativeError = (lossFromBeta != 0.0f) ? (lossError / ScalarAbs(lossFromBeta)) : lossError;
    
    // 打印 lossRelativeError
    {
        float val = lossRelativeError;
        int sign = (val < 0.0f);
        float absVal = sign ? -val : val;
        int exp = 0;
        if (absVal >= 1.0f) {
            while (absVal >= 10.0f) { absVal /= 10.0f; exp++; }
        } else if (absVal > 0.0f) {
            while (absVal < 1.0f) { absVal *= 10.0f; exp--; }
        }
        int digits = (int)(absVal * 1000000.0f + 0.5f);
        printf("lossRelErr: %s%d x10^%d\n", sign ? "-" : "", digits, exp - 6);
    }

    // 打印 lossError
    {
        float val = lossError;
        int sign = (val < 0.0f);
        float absVal = sign ? -val : val;
        int exp = 0;
        if (absVal >= 1.0f) {
            while (absVal >= 10.0f) { absVal /= 10.0f; exp++; }
        } else if (absVal > 0.0f) {
            while (absVal < 1.0f) { absVal *= 10.0f; exp--; }
        }
        int digits = (int)(absVal * 1000000.0f + 0.5f);
        printf("lossAbsErr: %s%d x10^%d\n", sign ? "-" : "", digits, exp - 6);
    }
        */
    pipe_.Reset();
}
#pragma endregion

#pragma region Beta
__aicore__ inline void OptimizedTransducer::ComputeBeta(int32_t sampleIdx)
{
    uint32_t uSizeAligned = (U_ + 7) / 8 * 8;
    uint32_t rowBytes = uSizeAligned * sizeof(float);

    // TPipe 空间规划:
    // - inQueue_: 概率行输入 (logPBlank + logPLabel = 2 * rowBytes)
    // - outQueue_: beta 输出
    // - workBuf_: targets + 2 rows beta + scalarLocal_
    uint32_t probRowBytes = 2 * rowBytes;
    uint32_t workBytes = 3 * rowBytes + scalarSize_;

    pipe_.InitBuffer(inQueue_, BUFFER_NUM, probRowBytes);
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, rowBytes);
    pipe_.InitBuffer(workBuf_, workBytes);

    scalarLocal_ = workBuf_.GetWithOffset<float>(scalarSize_ / sizeof(float), 3 * rowBytes);

    LocalTensor<int32_t> targetsLocal = workBuf_.Get<int32_t>(rowBytes / sizeof(int32_t));
    LocalTensor<float> betaRow[2];
    betaRow[0] = workBuf_.GetWithOffset<float>(uSizeAligned, rowBytes);
    betaRow[1] = workBuf_.GetWithOffset<float>(uSizeAligned, 2 * rowBytes);

    // 1. 加载 targets
    DataCopyExtParams targetCopyParams{1, static_cast<uint32_t>((U_ - 1) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> targetPadParams{false, 0, 0, 0};
    DataCopyPad(targetsLocal, targetsGm[sampleIdx * maxTargetLength_], targetCopyParams, targetPadParams);
    Sync<HardEvent::MTE2_S>();

    int32_t t_last = T_ - 1;

    // 2. 加载 t=T-1 行概率
    LocalTensor<float> probIn = inQueue_.AllocTensor<float>();
    LoadProbRow(t_last, uSizeAligned, targetsLocal, probIn);
    inQueue_.EnQue(probIn);

    // 3. 计算边界行 (t = T-1)
    probIn = inQueue_.DeQue<float>();
    LocalTensor<float> logPBlank = probIn;
    LocalTensor<float> logPLabel = probIn[uSizeAligned];

    int32_t currIdx = 0;
    // beta(T-1, U-1) = logPBlank(T-1, U-1)
    betaRow[currIdx].SetValue(U_ - 1, logPBlank.GetValue(U_ - 1));
    for (int32_t u = U_ - 2; u >= 0; --u) {
        // beta(T-1, u) = beta(T-1, u+1) + logPLabel(T-1, u)
        betaRow[currIdx].SetValue(u, betaRow[currIdx].GetValue(u + 1) + logPLabel.GetValue(u));
    }
    inQueue_.FreeTensor(probIn);

    // 输出边界行
    StoreBetaRow(t_last, uSizeAligned, betaRow[currIdx]);

    // 4. 主迭代 (t = T-2 down to 0)
    for (int32_t t = T_ - 2; t >= 0; --t) {
        int32_t nextIdx = 1 - currIdx;

        // 加载当前 t 行的概率 logPBlank logPLabel
        LocalTensor<float> probIn = inQueue_.AllocTensor<float>();
        LoadProbRow(t, uSizeAligned, targetsLocal, probIn);
        inQueue_.EnQue(probIn);
        probIn = inQueue_.DeQue<float>();
        LocalTensor<float> logPBlank = probIn;
        LocalTensor<float> logPLabel = probIn[uSizeAligned];

        // 边界: u = U-1, 只能向 t+1 方向转移
        betaRow[nextIdx].SetValue(U_ - 1, betaRow[currIdx].GetValue(U_ - 1) + logPBlank.GetValue(U_ - 1));

        // 其余位置: 两个方向的 LogAdd
        for (int32_t u = U_ - 2; u >= 0; --u) {
            // beta(t, u) = LogAdd(beta(t+1, u) + logPBlank(t, u), beta(t, u+1) + logPLabel(t, u))
            float betaToT = betaRow[currIdx].GetValue(u) + logPBlank.GetValue(u);
            float betaToU = betaRow[nextIdx].GetValue(u + 1) + logPLabel.GetValue(u);
            betaRow[nextIdx].SetValue(u, LogAdd(betaToT, betaToU));
        }
        inQueue_.FreeTensor(probIn);

        // 输出当前行
        StoreBetaRow(t, uSizeAligned, betaRow[nextIdx]);

        currIdx = nextIdx;
    }

    // 5. 提取并存储 loss
    cachedLoss_ = -betaRow[currIdx].GetValue(0);
    scalarLocal_.SetValue(0, cachedLoss_);
    pipe_barrier(PIPE_V);
    DataCopyExtParams lossCopyParams{1, sizeof(float), 0, 0, 0};
    DataCopyPad(lossGm[sampleIdx], scalarLocal_, lossCopyParams);

    pipe_.Reset();
}
#pragma endregion

#pragma region Grad

// 批量处理多个向量的梯度计算 (vocabSize_ < tileLen 时使用)
__aicore__ inline void OptimizedTransducer::ComputeGradForBatchVector(
    int64_t startVectorIdx, int32_t vectorCount, uint32_t tileLen, int32_t sampleIdx)
{
    int64_t gmOffset = (currentSampleOffset_ + startVectorIdx) * vocabSize_;

    GlobalTensor<float>& logProbGm = GetLogProbGm();
    float loss = cachedLoss_;
    bool lossInvalid = (loss > 1.0e20f || loss < -1.0e20f || (loss != loss));

    // Stage 1: Copy In
    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
    for (int32_t v = 0; v < vectorCount; ++v) {
        int64_t srcOffset = gmOffset + v * vocabSize_;
        uint32_t dstOffset = v * alignedVocabSize_;
        DataCopy(inLocal[dstOffset], logProbGm[srcOffset], alignedVocabSize_);
    }
    inQueue_.EnQue(inLocal);

    // Stage 2: Compute
    inLocal = inQueue_.DeQue<float>();
    LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();

    for (int32_t v = 0; v < vectorCount; ++v) {
        int64_t vectorIdx = startVectorIdx + v;
        int32_t t = static_cast<int32_t>(vectorIdx / U_);
        int32_t u = static_cast<int32_t>(vectorIdx % U_);

        uint32_t vOffset = v * alignedVocabSize_;
        LocalTensor<float> vInput = inLocal[vOffset];
        LocalTensor<float> vOutput = outLocal[vOffset];

        if (lossInvalid) {
            // loss 无效时，梯度全为 0
            Duplicate(vOutput, 0.0f, vocabSize_);
            continue;
        }

        // 计算当前位置的参数
        int64_t posIdx = currentSampleOffset_ + vectorIdx;
        float alpha = alphaGm.GetValue(posIdx);
        float beta = betaGm.GetValue(posIdx);
        float c = alpha + loss;
        float logGamma = c + beta;

        // 1. grad = exp(log_prob + logGamma)
        Adds(vOutput, vInput, logGamma, vocabSize_);
        Exp(vOutput, vOutput, vocabSize_);
        pipe_barrier(PIPE_V);

        // 2. 计算 gradBlankPosterior 并更新 blank 位置
        float logPBlank = vInput.GetValue(blankIdx_);
        float gBlank = logPBlank + c;
        float gradBlankPosterior = 0.0f;

        if (t == T_ - 1 && u == U_ - 1) {
            gradBlankPosterior = ScalarExp(gBlank);
        } else if (t < T_ - 1) {
            int64_t posNextT = currentSampleOffset_ + (t + 1) * U_ + u;
            float betaNextT = betaGm.GetValue(posNextT);
            gradBlankPosterior = ScalarExp(gBlank + betaNextT);
        }

        float currentGradBlank = vOutput.GetValue(blankIdx_);
        vOutput.SetValue(blankIdx_, currentGradBlank - gradBlankPosterior);

        // 3. 计算 gradLabelPosterior 并更新 label 位置
        int32_t targetLabel = (u < U_ - 1) ? GetTargetLabel(sampleIdx, u) : -1;
        if (targetLabel >= 0 && targetLabel < vocabSize_) {
            float logPLabel = vInput.GetValue(targetLabel);
            float gLabel = logPLabel + c;
            int64_t posNextU = currentSampleOffset_ + t * U_ + (u + 1);
            float betaNextU = betaGm.GetValue(posNextU);
            float gradLabelPosterior = ScalarExp(gLabel + betaNextU);

            float currentGradLabel = vOutput.GetValue(targetLabel);
            vOutput.SetValue(targetLabel, currentGradLabel - gradLabelPosterior);
        }

        // 4. Clamp
        if (clamp_ > 0.0f) {
            Maxs(vOutput, vOutput, -clamp_, vocabSize_);
            Mins(vOutput, vOutput, clamp_, vocabSize_);
        }
    }

    outQueue_.EnQue(outLocal);
    inQueue_.FreeTensor(inLocal);

    // Stage 3: Copy Out
    outLocal = outQueue_.DeQue<float>();
    for (int32_t v = 0; v < vectorCount; ++v) {
        uint32_t srcOffset = v * alignedVocabSize_;
        int64_t dstOffset = gmOffset + v * vocabSize_;
        uint32_t copySize = vocabSize_ * sizeof(float);

        DataCopyExtParams copyOutParams{1, copySize, 0, 0, 0};
        DataCopyPad(gradGm[dstOffset], outLocal[srcOffset], copyOutParams);
    }
    outQueue_.FreeTensor(outLocal);
}

// 处理单个大向量的梯度计算 (vocabSize_ >= tileLen 时使用)
__aicore__ inline void OptimizedTransducer::ComputeGradSingleLargeVector(
    int64_t vectorIdx, uint32_t tileLen, float logGamma, float gradBlankPosterior, float gradLabelPosterior,
    int32_t targetLabel)
{
    int64_t gmOffset = (currentSampleOffset_ + vectorIdx) * vocabSize_;
    uint32_t loopCount = (vocabSize_ + tileLen - 1) / tileLen;

    GlobalTensor<float>& logProbGm = GetLogProbGm();

    for (uint32_t i = 0; i < loopCount; ++i) {
        int32_t curOffset = i * tileLen;
        int32_t curLen = (i == loopCount - 1) ? (vocabSize_ - curOffset) : tileLen;
        uint32_t copySize = curLen * sizeof(float);

        // Stage 1: Copy In
        LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
        DataCopyExtParams copyParams{1, copySize, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0.0f};
        DataCopyPad(inLocal, logProbGm[gmOffset + curOffset], copyParams, padParams);
        inQueue_.EnQue(inLocal);

        // Stage 2: Compute
        inLocal = inQueue_.DeQue<float>();
        LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();

        // grad = exp(log_prob + logGamma)
        Adds(outLocal, inLocal, logGamma, curLen);
        Exp(outLocal, outLocal, curLen);
        pipe_barrier(PIPE_V);

        // Window Check for Blank
        int32_t startIdx = curOffset;
        int32_t endIdx = curOffset + curLen;

        if (blankIdx_ >= startIdx && blankIdx_ < endIdx) {
            int32_t localIdx = blankIdx_ - startIdx;
            float currentGradBlank = outLocal.GetValue(localIdx);
            outLocal.SetValue(localIdx, currentGradBlank - gradBlankPosterior);
        }

        // Window Check for Label
        if (targetLabel >= 0 && targetLabel >= startIdx && targetLabel < endIdx) {
            int32_t localIdx = targetLabel - startIdx;
            float currentGradLabel = outLocal.GetValue(localIdx);
            outLocal.SetValue(localIdx, currentGradLabel - gradLabelPosterior);
        }

        // Clamp
        if (clamp_ > 0.0f) {
            Maxs(outLocal, outLocal, -clamp_, curLen);
            Mins(outLocal, outLocal, clamp_, curLen);
        }

        outQueue_.EnQue(outLocal);
        inQueue_.FreeTensor(inLocal);

        // Stage 3: Copy Out
        outLocal = outQueue_.DeQue<float>();
        DataCopyPad(gradGm[gmOffset + curOffset], outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }
}

__aicore__ inline void OptimizedTransducer::ComputeGrad(int32_t sampleIdx)
{
    int64_t totalPos = static_cast<int64_t>(T_) * U_;
    uint32_t bufferSize = (ubSize_ - scalarSize_) / 5 / 32 * 32;
    uint32_t tileLen = bufferSize / sizeof(float);

    // 初始化 TPipe
    pipe_.InitBuffer(inQueue_, BUFFER_NUM, bufferSize);  // 2
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, bufferSize);  // 2
    pipe_.InitBuffer(workBuf_, bufferSize + scalarSize_);  // 1 + scalarLocal_

    // 分配 scalarLocal_
    scalarLocal_ = workBuf_.GetWithOffset<float>(scalarSize_ / sizeof(float), bufferSize);

    GlobalTensor<float>& logProbGm = GetLogProbGm();
    float loss = cachedLoss_;
    bool lossInvalid = (loss > 1.0e20f || loss < -1.0e20f || (loss != loss));

    if (alignedVocabSize_ <= tileLen) {
        // 批量处理模式：一次处理多个向量
        uint32_t vectorsPerBatch = tileLen / alignedVocabSize_;
        for (int64_t i = 0; i < totalPos; i += vectorsPerBatch) {
            uint32_t currentBatchSize = (i + vectorsPerBatch > totalPos) ? (totalPos - i) : vectorsPerBatch;
            ComputeGradForBatchVector(i, currentBatchSize, tileLen, sampleIdx);
        }
    } else {
        // 大向量模式：逐个处理
        for (int64_t i = 0; i < totalPos; ++i) {
            int32_t t = static_cast<int32_t>(i / U_);
            int32_t u = static_cast<int32_t>(i % U_);
            int64_t posIdx = currentSampleOffset_ + i;

            if (lossInvalid) {
                ComputeGradSingleLargeVector(i, tileLen, -1.0e20f, 0.0f, 0.0f, -1);
                continue;
            }

            // 计算当前位置的参数
            float alpha = alphaGm.GetValue(posIdx);
            float beta = betaGm.GetValue(posIdx);
            Sync<HardEvent::MTE2_S>();
            float c = alpha + loss;
            float logGamma = c + beta;

            int64_t offset = GetPosOffsetLocal(t, u);
            float logPBlank = logProbGm.GetValue(offset + blankIdx_);
            float gBlank = logPBlank + c;

            float gradBlankPosterior = 0.0f;
            if (t == T_ - 1 && u == U_ - 1) {
                gradBlankPosterior = ScalarExp(gBlank);
            } else if (t < T_ - 1) {
                int64_t posNextT = currentSampleOffset_ + (t + 1) * U_ + u;
                float betaNextT = betaGm.GetValue(posNextT);
                Sync<HardEvent::MTE2_S>();
                gradBlankPosterior = ScalarExp(gBlank + betaNextT);
            }

            int32_t targetLabel = (u < U_ - 1) ? GetTargetLabel(sampleIdx, u) : -1;
            float gradLabelPosterior = 0.0f;
            if (targetLabel >= 0 && targetLabel < vocabSize_) {
                float logPLabel = logProbGm.GetValue(offset + targetLabel);
                float gLabel = logPLabel + c;
                int64_t posNextU = currentSampleOffset_ + t * U_ + (u + 1);
                float betaNextU = betaGm.GetValue(posNextU);
                gradLabelPosterior = ScalarExp(gLabel + betaNextU);
            }

            ComputeGradSingleLargeVector(i, tileLen, logGamma, gradBlankPosterior, gradLabelPosterior, targetLabel);
        }
    }

    pipe_.Reset();
}
#pragma endregion

#pragma region Main_Process
__aicore__ inline void OptimizedTransducer::Process()
{
    for (int32_t sample = sampleStart_; sample < sampleEnd_; ++sample) {
        GetCurrentTU(sample, T_, U_);

        ComputeLogSoftmax(sample);

        ComputeBeta(sample);

         if (clamp_ != -2.0f) {
             ComputeAlpha(sample);
             ComputeGrad(sample);
         }

        currentSampleOffset_ += static_cast<int64_t>(T_) * U_;
    }
}
#pragma endregion

} // namespace NsOptimizedTransducer
#endif // __OPTIMIZED_TRANSDUCER_H__