#include "kernel_operator.h"
#include "stdio.h"
using namespace AscendC;


namespace NsUniqueV3
{

template<typename T>
__aicore__ inline void KernelUnique<T>::ConsecutiveUnique(
    const LocalTensor<float>& dstVal,
    const LocalTensor<float>& srcLocal,
    const LocalTensor<float>& shiftedLocal,
    const LocalTensor<uint32_t>& bitMask32,
    const uint16_t elemLength,
    uint64_t& tileUniqueCnt)
{
    uint64_t rsvdCnt = 0;
    GatherMask(dstVal, srcLocal, 1, false, 0, 
        {1, static_cast<uint16_t>((elemLength * 2 + 63) / 64), 8, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // 构造左移 mask：最低位 0，其余全 1
    Duplicate(bitMask32, (uint32_t)0xFFFFFFFF, (elemLength + 31) / 32);
    PipeBarrier<PIPE_V>();
    bitMask32.SetValue(0, 0xFFFFFFFE);
    // 左移：shiftedLocal[i] = dstVal[i+1]
    GatherMask(shiftedLocal, dstVal, bitMask32, true, elemLength, {1, 1, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // 末尾补 -INF，让最后一个有效值被识别为"下一个是-INF所以保留"
    shiftedLocal.SetValue(elemLength - 1, -FLOAT_INF);
    // NE 比较：dstVal[i] != shiftedLocal[i] （= dstVal[i+1]）
    LocalTensor<uint32_t> neMask = bitMask32[TILE_LENGTH / 2].ReinterpretCast<uint32_t>();
    LocalTensor<uint8_t> neMask8 = neMask.ReinterpretCast<uint8_t>();
    Compare(neMask8, dstVal, shiftedLocal, CMPMODE::NE, elemLength);  // ← 直接用 elemLength，TILE_LENGTH 天然 64 对齐
    PipeBarrier<PIPE_V>();
    // 按 mask 收集"段末尾"值，padding 区的 -INF != -INF = 0，天然不会被收集
    GatherMask(shiftedLocal, dstVal, neMask, true, elemLength, {1, 1, 8, 8}, tileUniqueCnt);
    PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline void KernelUnique<T>::CalculateUnique()
{
    float lastValue = FLOAT_INF;
    uint32_t localUniqueCnt = 0;
    float firstUniqueVal = FLOAT_INF;

    for (int32_t tileIdx = 0; tileIdx < (int32_t)this->tileNum; tileIdx++) {
        LocalTensor<uint32_t> bitMask32 = calcBuf[0].Get<uint32_t>();
        LocalTensor<float> shiftedLocal = bitMask32[TILE_LENGTH].ReinterpretCast<float>();
        LocalTensor<float> sortedLocal = calcBuf[1].Get<float>();
        LocalTensor<float> dstVal = calcBuf[2].Get<float>();

        AscendC::DataCopyPad(sortedLocal, sortedBlock1[tileIdx * TILE_LEN_ELEM], 
            {1, static_cast<uint32_t>(TILE_LEN_ELEM * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();

        uint64_t tileUniqueCnt = 0;
        ConsecutiveUnique(dstVal, sortedLocal, shiftedLocal, bitMask32, TILE_LENGTH, tileUniqueCnt);
        
        bool skipFirst = (tileUniqueCnt > 0 && shiftedLocal.GetValue(0) == lastValue);
        if (tileUniqueCnt > 0) {
            lastValue = shiftedLocal.GetValue(tileUniqueCnt - 1);
        }

        uint32_t writeLen = (uint32_t)tileUniqueCnt;

        if (localUniqueCnt == 0 && writeLen > 0) {
            firstUniqueVal = shiftedLocal.GetValue(skipFirst ? 1 : 0);
        }

        if (writeLen > 0) {
            // GM 地址往前退一位，用 shiftedLocal[0]（对齐）直接覆盖上一个 tile 的重复尾值
            uint32_t gmOff = skipFirst ? (localUniqueCnt - 1) : localUniqueCnt;
            SyncDiffPipe<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(sortedBlock2[gmOff], shiftedLocal,
                        {1, static_cast<uint32_t>(sizeof(float) * writeLen), 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            localUniqueCnt = gmOff + writeLen;
        }
    }

    LocalTensor<float> tmpLocal = calcBuf[1].Get<float>();
    tmpLocal.SetValue(0, firstUniqueVal);
    tmpLocal.SetValue(1, lastValue);
    tmpLocal.ReinterpretCast<uint32_t>().SetValue(2, localUniqueCnt);
    SyncDiffPipe<AscendC::HardEvent::S_MTE3>();       
    DataCopyPad(uniqueMsg[GetBlockIdx() * 3], tmpLocal,
        {1, static_cast<uint32_t>(sizeof(float) * 3), 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
}

template<typename T>
__aicore__ inline void KernelUnique<T>::CopyOutUnique()
{
    LocalTensor<float> tmpLocal = calcBuf[1].Get<float>();
    DataCopyPad(tmpLocal, uniqueMsg,
        {1, static_cast<uint32_t>(sizeof(float) * 3 * blockNum), 0, 0, 0},
        {false, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();

    uint32_t writeOffset = 0;
    float prevLast = -FLOAT_INF;
    bool mySkipFirst = false;
    uint32_t myCountLen = 0;
    uint32_t totalUniqueCnt = 0;

    for (int32_t i = 0; i < (int32_t)blockNum; i++) {
        float first_i = tmpLocal.GetValue(i * 3);
        float last_i = tmpLocal.GetValue(i * 3 + 1);
        uint32_t count_i = tmpLocal.ReinterpretCast<uint32_t>().GetValue(i * 3 + 2);

        bool skip = (i > 0 && count_i > 0 && first_i == prevLast);
        uint32_t effective = count_i - (skip ? 1 : 0);

        if (i == (int32_t)GetBlockIdx()) {
            mySkipFirst = skip;
            myCountLen = count_i;
        }
        if (i < (int32_t)GetBlockIdx()) {
            writeOffset += effective;
        }
        totalUniqueCnt += effective;

        if (count_i > 0) {
            prevLast = last_i;
        }
    }

    uint32_t srcOff = mySkipFirst ? 1 : 0;
    uint32_t writeLen = myCountLen - srcOff;
    if (writeLen > 0) {
        NsUniqueV3::DataCopyGM2GMCast(
            dstGlobal[writeOffset],                          
            sortedBlock2[srcOff],                             
            calcBuf[0].template Get<float>(),                 
            calcBuf[1].template Get<T>(), 
            writeLen,
            TILE_LEN_BYTE);
        PipeBarrier<PIPE_ALL>();
    }
    // 写出最终的unique count
    if (GetBlockIdx() == 0) {
        uniqueCntGlobal.SetValue(0, totalUniqueCnt);
    }
}

}