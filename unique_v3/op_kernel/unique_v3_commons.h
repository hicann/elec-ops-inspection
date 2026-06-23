#ifndef __UNIQUE_V3_TOOLS_H__
#define __UNIQUE_V3_TOOLS_H__

#include "kernel_operator.h"



#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


namespace NsUniqueV3 {

template <AscendC::HardEvent event>
__aicore__ inline void SyncDiffPipe()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

template<typename T>
__aicore__ inline void DataCopyGM2GM(const AscendC::GlobalTensor<T>& dst, 
    const AscendC::GlobalTensor<T>& src,
    const AscendC::LocalTensor<T>& tmpLocal, 
    const int elemLength, const int bufByteLength)
{
    int bufElemLength = bufByteLength / sizeof(T);
    int restLen = elemLength;
    while (restLen > 0) {
        int copyLen = MIN(restLen, bufElemLength);
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T) * copyLen), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(tmpLocal, src[elemLength - restLen], params, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(dst[elemLength - restLen], tmpLocal, params);
        AscendC::PipeBarrier<PIPE_ALL>();
        restLen -= copyLen;
    }
}


template<typename T1, typename T2>
__aicore__ inline void DataCopyGM2GMCast(const AscendC::GlobalTensor<T2>& dst,
    const AscendC::GlobalTensor<T1>& src,
    const AscendC::LocalTensor<T1>& tmpLocal1,
    const AscendC::LocalTensor<T2>& tmpLocal2,
    const int elemLength, const int bufByteLength)
{
    int bufElemLength = bufByteLength / (sizeof(T1) > sizeof(T2) ? sizeof(T1) : sizeof(T2));
    int maxPerCopy1 = 65535 / sizeof(T1);
    int maxPerCopy2 = 65535 / sizeof(T2);
    if (bufElemLength > maxPerCopy1) bufElemLength = maxPerCopy1;
    if (bufElemLength > maxPerCopy2) bufElemLength = maxPerCopy2;

    int restLen = elemLength;
    while (restLen > 0) {
        int copyLen = MIN(restLen, bufElemLength);

        AscendC::DataCopyExtParams srcParams{1, static_cast<uint32_t>(sizeof(T1) * copyLen), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T1> srcPadParams{false, 0, 0, 0};
        AscendC::DataCopyPad(tmpLocal1, src[elemLength - restLen], srcParams, srcPadParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        if constexpr (AscendC::IsSameType<T1, T2>::value) {
            // 同类型:直接写,不需要 Cast
            AscendC::DataCopyExtParams dstParams{1, static_cast<uint32_t>(sizeof(T2) * copyLen), 0, 0, 0};
            AscendC::DataCopyPad(dst[elemLength - restLen], tmpLocal1.template ReinterpretCast<T2>(), dstParams);
        } else {
            // Cast 模式选择:
            //   float → int (缩小精度/类型变换): CAST_RINT (round to nearest integer)
            //   int → float (升精度): CAST_ROUND (带舍入,避免 int32 超出 float 精度时编译器警告)
            //   half/bf16 → float (升精度,无损): CAST_NONE
            //   float → half/bf16 (降精度): CAST_RINT
            if constexpr (AscendC::IsSameType<T1, float>::value && sizeof(T2) >= sizeof(float)) {
                // float → int32: 四舍五入到最近整数
                AscendC::Cast(tmpLocal2, tmpLocal1, AscendC::RoundMode::CAST_RINT, copyLen);
            } else if constexpr (sizeof(T1) >= sizeof(float) && AscendC::IsSameType<T2, float>::value) {
                // int32 → float: 带舍入
                AscendC::Cast(tmpLocal2, tmpLocal1, AscendC::RoundMode::CAST_ROUND, copyLen);
            } else if constexpr (sizeof(T1) < sizeof(float) && AscendC::IsSameType<T2, float>::value) {
                // half/bf16/int16 → float: 无损升精度
                AscendC::Cast(tmpLocal2, tmpLocal1, AscendC::RoundMode::CAST_NONE, copyLen);
            } else {
                // 其他组合(如 float → half/bf16): 四舍五入
                AscendC::Cast(tmpLocal2, tmpLocal1, AscendC::RoundMode::CAST_RINT, copyLen);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopyExtParams dstParams{1, static_cast<uint32_t>(sizeof(T2) * copyLen), 0, 0, 0};
            AscendC::DataCopyPad(dst[elemLength - restLen], tmpLocal2, dstParams);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        restLen -= copyLen;
    }
}

}

#endif