

#ifndef __UNIQUE_V3_H__
#define __UNIQUE_V3_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "unique_v3_tiling_data.h"
#include "unique_v3_tiling_key.h"
#include "unique_v3_commons.h"


namespace NsUniqueV3 {

using namespace AscendC;



template<typename T>
class KernelUnique {
public:
    __aicore__ inline KernelUnique(TPipe& pipe) : pipe(pipe) {}
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt,
        GM_ADDR inverse, GM_ADDR counts, GM_ADDR workspace,
        const uint32_t totalLength, const uint32_t shortBlockTileNum, const uint16_t tileLength,
        const uint16_t tailLength, const uint8_t aivNum, const uint8_t blockNum, const uint8_t shortBlockNum,
        const bool flagSorted, const bool flagInverse, const bool flagCounts);
    __aicore__ inline void Process();
    __aicore__ inline size_t GetGlobalOffset(const uint32_t blockIdx);


private:

    using GMSSrcList = GlobalTensor<float> (&)[4];
    struct GMSParams {
        int (&GMSLengths)[4];
        uint8_t& queNum;
        LocalTensor<float> (&&buffLocal)[5];
    };

    __aicore__ inline void SortTile();
    __aicore__ inline bool MrgTile(const LocalTensor<float>& sortArray,
                                   const LocalTensor<float>& tmpArray,
                                   int32_t tileLen);
    __aicore__ inline void MrgSortGM(GlobalTensor<float>&& dstGlobal, 
                                    GMSSrcList& srcList, 
                                    GMSParams& params);                                   
    __aicore__ inline void MrgBlock();
    __aicore__ inline void MrgGlobal();

    __aicore__ inline void CalculateFlip();
    __aicore__ inline void CalculateUnique();
    __aicore__ inline void CalculateInverse();
    __aicore__ inline void CalculateCounts();

    

    __aicore__ inline void CopyOriginalArrayIdx2GM(
        const LocalTensor<float> &ArrayLocal, const LocalTensor<float> &idxLocal,
        const LocalTensor<uint32_t> &tmpLocal, int32_t progress);
    __aicore__ inline void TileCumulativeSum(const LocalTensor<float> &sortedLocal1,
        const LocalTensor<float> &sortedLocal2, const LocalTensor<uint32_t>& tmpLocal,
        int32_t progress, int32_t &unique_num, float &firstValue, float &endValue);
    __aicore__ inline void BlockCumulativeSum();

    __aicore__ inline static bool TileCalculateCounts(const LocalTensor<float>& dstVal,
        const LocalTensor<float>& srcLocal, const LocalTensor<float>& shiftedLocal,
        const LocalTensor<uint32_t>& bitMask32, const uint16_t elemLength,
        uint64_t& arrayLen,int32_t& beforeNumCnt, float& beforeNumValue);
    __aicore__ inline static void ConsecutiveUnique(const LocalTensor<float>& dstVal,
        const LocalTensor<float>& srcLocal, const LocalTensor<float>& shiftedLocal,
        const LocalTensor<uint32_t>& bitMask16, const uint16_t elemLength, uint64_t& tileUniqueCnt);
    __aicore__ inline void TileUnique(const int32_t progress);

    __aicore__ inline void CopyOutUnique();
    __aicore__ inline void CopyOutCounts();
    __aicore__ inline void CopyOutInverse();
    __aicore__ inline void CopyOut();






private:
    static constexpr int32_t TILE_LENGTH = 8192;
    // INF to fill the tail blank, so that tail is automatically removed by Compare in Unique.
    static constexpr float FLOAT_INF = 3.402823e+38f;
    // Indicates the factor converting float to data structure used by Sort32&MrgSort.
    static constexpr int16_t SORT_DATATYPE_SIZE = sizeof(float) + sizeof(uint32_t);          // 8
    static constexpr int16_t SORT_DATATYPE_SIZE_FACTOR = SORT_DATATYPE_SIZE / sizeof(float); // 2
    static constexpr int32_t TILE_LEN_BYTE = TILE_LENGTH * SORT_DATATYPE_SIZE;               // 8192 * 8 = 65536
    static constexpr int32_t TILE_LEN_ELEM = TILE_LENGTH * SORT_DATATYPE_SIZE_FACTOR;        // 8192 * 2 = 16384
    // Max elements per way for MrgSort, limited by UB output buffer (8192 elements total)
    // 4-way: 8192/4=2048, but cap at 2047 to keep total bytes < 65535 for DataCopyPad
    // 3-way: 8190/3=2730, 2-way: min(4095, 8190/2)=4095
    static constexpr int32_t BUFFER_LEN[5] = {0, 0, 4095, 2730, 2048};
    static constexpr uint16_t VALID_QUE[5] = {0, 0, 0b11, 0b111, 0b1111};

    AscendC::TPipe& pipe;
    TBuf<TPosition::VECCALC> calcBuf[3];

    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> srcBlock;
    GlobalTensor<T> dstGlobal;
    GlobalTensor<int32_t> uniqueCntGlobal;
    GlobalTensor<int32_t> counterResult;
    GlobalTensor<int32_t> inverseResult;
    GlobalTensor<int32_t> inverseResultBlock;

    GlobalTensor<float> sortedGlobal1;
    GlobalTensor<float> sortedGlobal2;
    GlobalTensor<float> sortedBlock1;
    GlobalTensor<float> sortedBlock2;
    GlobalTensor<int32_t> sortedBlock1AsInt32;
    GlobalTensor<int32_t> sortedBlock2AsInt32; 

    GlobalTensor<int32_t> IBSyncGlobal;
    GlobalTensor<float> uniqueMsg;

    GlobalTensor<int32_t> counterGlobal;
    GlobalTensor<float> counterMsg;
    GlobalTensor<int32_t> inverseGlobal1;
    GlobalTensor<int32_t> inverseGlobal2;
    GlobalTensor<int32_t> inverseBlock1;
    GlobalTensor<int32_t> inverseBlock2;
    GlobalTensor<float> inverseMsg;

    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t shortBlockTileNum;
    uint16_t tailLength;
    uint16_t syncWorkspaceSize;
    uint8_t blockNum;
    uint8_t shortBlockNum;
    size_t globalOffset;
    size_t blockLength;
    size_t blockRealLength;


    uint8_t eventID{0};
    bool hasInfFlag {false};
    bool flagSorted{true};
    bool flagInverse {false};
    bool flagCounts {false};

};



template<typename T>
__aicore__ inline void KernelUnique<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt,
    GM_ADDR inverse, GM_ADDR counts, GM_ADDR workspace,
    const uint32_t totalLength, const uint32_t shortBlockTileNum, const uint16_t tileLength,
    const uint16_t tailLength, const uint8_t aivNum, const uint8_t blockNum, const uint8_t shortBlockNum,
    const bool flagSorted, const bool flagInverse, const bool flagCounts)
{
    this->totalLength = totalLength;
    this->shortBlockTileNum = shortBlockTileNum;
    this->tailLength = tailLength;
    this->blockNum = blockNum;
    this->shortBlockNum = shortBlockNum;
    this->flagSorted = flagSorted;
    this->flagInverse = flagInverse;
    this->flagCounts = flagCounts;

    uint32_t alignedTotalLength = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH * TILE_LENGTH;
    const bool isShortBlock = this->shortBlockNum > GetBlockIdx();
    this->tileNum = isShortBlock ? shortBlockTileNum : shortBlockTileNum + 1;
    this->blockLength = this->tileNum * TILE_LENGTH;
    this->globalOffset = GetGlobalOffset(GetBlockIdx());
    this->syncWorkspaceSize = (blockNum * 32 * 8 + aivNum * 32 + 32) / sizeof(int32_t);
    this->blockRealLength = MIN((int32_t)blockLength, (int32_t)totalLength - (int32_t)globalOffset);

    // 初始化输入及输出 GM空间
    srcGlobal.SetGlobalBuffer((__gm__ T*)input, alignedTotalLength);
    srcBlock.SetGlobalBuffer((__gm__ T*)input + globalOffset, this->blockRealLength);
    dstGlobal.SetGlobalBuffer((__gm__ T*)output, alignedTotalLength);
    uniqueCntGlobal.SetGlobalBuffer((__gm__ int32_t*)uniqueCnt, 1);
    inverseResult.SetGlobalBuffer((__gm__ int32_t*)inverse, alignedTotalLength);
    inverseResultBlock.SetGlobalBuffer((__gm__ int32_t*)inverse + globalOffset, this->blockLength);
    counterResult.SetGlobalBuffer((__gm__ int32_t*)counts, alignedTotalLength);
    
    // 初始化unique(核内及核间ping-pong归并) GM临时空间
    sortedGlobal1.SetGlobalBuffer((__gm__ float*)workspace, alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedGlobal2.SetGlobalBuffer((__gm__ float*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR, alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedBlock1.SetGlobalBuffer((__gm__ float*)workspace + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedBlock2.SetGlobalBuffer((__gm__ float*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedBlock1AsInt32.SetGlobalBuffer((__gm__ int32_t*)workspace + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    sortedBlock2AsInt32.SetGlobalBuffer((__gm__ int32_t*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);

    // 初始化核间同步 GM临时空间
    IBSyncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR * 2, syncWorkspaceSize);

    // 初始化unique的核间同步计数空间
    uniqueMsg.SetGlobalBuffer((__gm__ float*)workspace + alignedTotalLength * SORT_DATATYPE_SIZE_FACTOR * 2 + syncWorkspaceSize, ((blockNum + 7) / 8 * 8) * 3); 

    // 初始化counter及inverse GM临时空间
    uint32_t counterOffset = alignedTotalLength * 4 + syncWorkspaceSize + ((blockNum + 7) / 8 * 8) * 3;
    uint32_t inverstOffset = counterOffset + alignedTotalLength + ((blockNum + 7) / 8 * 8) * 3;
    counterGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace + counterOffset, alignedTotalLength);
    counterMsg.SetGlobalBuffer((__gm__ float*)workspace + counterOffset + alignedTotalLength, ((blockNum + 7) / 8 * 8) * 3);
    inverseGlobal1.SetGlobalBuffer((__gm__ int32_t*)workspace + inverstOffset, alignedTotalLength * 2);
    inverseGlobal2.SetGlobalBuffer((__gm__ int32_t*)workspace + inverstOffset + alignedTotalLength * 2, alignedTotalLength * 2);
    inverseBlock1.SetGlobalBuffer((__gm__ int32_t*)workspace + inverstOffset + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    inverseBlock2.SetGlobalBuffer((__gm__ int32_t*)workspace + inverstOffset + alignedTotalLength * 2 + globalOffset * SORT_DATATYPE_SIZE_FACTOR, this->blockLength * SORT_DATATYPE_SIZE_FACTOR);
    inverseMsg.SetGlobalBuffer((__gm__ float*)workspace + inverstOffset + alignedTotalLength * 4, ((blockNum + 7) / 8 * 8) * 3);

    if (blockNum > 1) {
        if (GetBlockIdx() == 0) {
            InitGlobalMemory(IBSyncGlobal, syncWorkspaceSize, 0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 初始化UB计算临时空间
    pipe.InitBuffer(calcBuf[0], TILE_LEN_BYTE);
    pipe.InitBuffer(calcBuf[1], TILE_LEN_BYTE);
    pipe.InitBuffer(calcBuf[2], TILE_LEN_BYTE);
}

template<typename T>
__aicore__ inline size_t KernelUnique<T>::GetGlobalOffset(const uint32_t blockIdx)
{
    const size_t offset =
        (this->shortBlockTileNum * MIN(this->shortBlockNum, blockIdx) +
            (this->shortBlockTileNum + 1) * (this->shortBlockNum >= blockIdx ? 0 : blockIdx - this->shortBlockNum)) * TILE_LENGTH;
    return offset;
}

template<typename T>
__aicore__ inline void KernelUnique<T>::Process()
{
    // 逐tile排序
    SortTile();
    // 核内归并
    MrgBlock();
    // 核间归并
    SyncAll();
    MrgGlobal();
    SyncAll();

    // 如果是递增输出，则需要翻转回去
    if(flagSorted) CalculateFlip();
    // counts计算
    if (flagCounts) CalculateCounts();
    // inverse计算
    if (flagInverse) CalculateInverse();
    // 去重
    CalculateUnique();
    SyncAll();

    // 结果写出
    CopyOut();
}

template <typename T>
__aicore__ inline void KernelUnique<T>::SortTile()
{
    LocalTensor<float> input = calcBuf[0].Get<float>();
    LocalTensor<float> tmp = input[TILE_LENGTH].ReinterpretCast<float>();
    LocalTensor<int32_t> arange = calcBuf[1].Get<int32_t>();
    
    for (uint32_t i = 0; i < tileNum; i++) {
        int32_t tileLen = MIN(TILE_LENGTH, blockRealLength - i * TILE_LENGTH);
        uint32_t repeat = (tileLen + 31) / 32;
        AscendC::Duplicate<float>(input, -FLOAT_INF, TILE_LENGTH);

        // 这里做多类型支持，先把输入转换成float，中间计算过程都用float进行，最后再转换回去。
         if constexpr (IsSameType<T, float>::value) {
            // float: 直接搬到 float buf
            AscendC::DataCopyPad(tmp, srcBlock[i * TILE_LENGTH], 
                {1, static_cast<uint32_t>(tileLen * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            // 非 float (int32 & fp16 bf16 int8等): 先读进来 再 Cast 成 float
            LocalTensor<T> srcAsT = calcBuf[2].Get<T>();
            AscendC::DataCopyPad(srcAsT, srcBlock[i * TILE_LENGTH],
                {1, static_cast<uint32_t>(tileLen * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            if constexpr (sizeof(T) >= sizeof(float)) {
                AscendC::Cast(tmp, srcAsT, AscendC::RoundMode::CAST_ROUND, tileLen);
            } else {
                AscendC::Cast(tmp, srcAsT, AscendC::RoundMode::CAST_NONE, tileLen);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        //如果需要递增输出，则乘以-1
        AscendC::Muls(input, tmp, flagSorted ? -1.0f : 1.0f, tileLen);
        AscendC::PipeBarrier<PIPE_V>();
	    //构造递增数组用于后续inverse计算
        AscendC::Arange(arange, static_cast<int32_t>(globalOffset + i * TILE_LENGTH), 1, TILE_LENGTH);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<float> dstLocal = calcBuf[2].Get<float>();
	    //255个repeat超限 拆成两次排
        AscendC::Sort32<float>(dstLocal, input, arange.ReinterpretCast<uint32_t>(), 128);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sort32<float>(dstLocal[TILE_LENGTH], input[TILE_LENGTH / 2], arange[TILE_LENGTH / 2].ReinterpretCast<uint32_t>(), 128);
        AscendC::PipeBarrier<PIPE_V>();
        bool readFromSort = MrgTile(dstLocal, input, TILE_LENGTH);
        SyncDiffPipe<AscendC::HardEvent::V_MTE3>();
        //tile内排序完成后写入GM
        AscendC::DataCopyPad(sortedBlock1[i * TILE_LENGTH * 2],
                            readFromSort ? dstLocal : input,
                            {2, static_cast<uint16_t>(sizeof(float) * TILE_LENGTH), 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline bool KernelUnique<T>::MrgTile(
    const LocalTensor<float>& sortArray,
    const LocalTensor<float>& tmpArray,
    int32_t numElements)
{
    int32_t numGroups = (numElements + 31) / 32;
    int32_t groupSize = 32;
    bool readFromSort = true;
    // tile内合并 调用MrgSort 以 4x32 -> 4x128 -> 4x512推进
    while (numGroups > 1) {
        // UB间ping-pong互换归并
        const LocalTensor<float>& src = readFromSort ? sortArray : tmpArray;
        const LocalTensor<float>& dst = readFromSort ? tmpArray : sortArray;
        int32_t stride = groupSize * 2;

        AscendC::MrgSortSrcList<float> srcList;
        AscendC::MrgSort4Info params;
        params.ifExhaustedSuspension = false;
        params.repeatTimes = 1;
        int32_t sets = (numGroups + 3) / 4;
        for (int s = 0; s < sets; s++) {
            int base = s * 4 * stride;
            int offset0 = base;
            int offset1 = base + stride;
            int offset2 = base + stride * 2;
            int offset3 = base + stride * 3;
            params.elementLengths[0] = (uint16_t)MIN(groupSize, MAX(0, numElements - offset0 / 2));
            params.elementLengths[1] = (uint16_t)MIN(groupSize, MAX(0, numElements - offset1 / 2));
            params.elementLengths[2] = (uint16_t)MIN(groupSize, MAX(0, numElements - offset2 / 2));
            params.elementLengths[3] = (uint16_t)MIN(groupSize, MAX(0, numElements - offset3 / 2));
            if (params.elementLengths[1] == 0) {
                Copy(dst[base], src[base], (uint64_t)64,
                     (uint8_t)((params.elementLengths[0] * 2 * 4 + 255) / 256),
                     {1, 1, 8, 8});
                AscendC::PipeBarrier<PIPE_ALL>();
                break;
            }
            params.validBit =
                (params.elementLengths[2] == 0 ? 3 :
                (params.elementLengths[3] == 0 ? 7 : 15));
            srcList.src1 = src[offset0];
            srcList.src2 = src[offset1];
            srcList.src3 = src[offset2];
            srcList.src4 = src[offset3];
            AscendC::MrgSort<float>(dst[base], srcList, params);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        numGroups = (numGroups + 3) / 4;
        groupSize *= 4;
        readFromSort = !readFromSort;
    }
    return readFromSort;
}

template <typename T>
__aicore__ inline void KernelUnique<T>::MrgSortGM(
    GlobalTensor<float>&& dstGlobal, GMSSrcList& srcList, GMSParams& params)
{
    int restLen[4] = {params.GMSLengths[0], params.GMSLengths[1], params.GMSLengths[2], params.GMSLengths[3]};
    int currentHead[4] = {0, 0, 0, 0};
    int totalMrgLen = 0;
    uint8_t queNum = params.queNum;
    uint16_t sortedLen[4];
    uint16_t mrgLen[4] = {0, 0, 0, 0};

    while (queNum > 1) {
        int currentBufferLen = BUFFER_LEN[queNum];
        for (int i = 0; i < queNum; i++) {
            mrgLen[i] = MIN(restLen[i], currentBufferLen);
        }
        // CopyIn
        for (int i = 0; i < queNum; i++) {
            AscendC::DataCopyPad(params.buffLocal[i],
                srcList[i][currentHead[i] * SORT_DATATYPE_SIZE_FACTOR],
                {1, static_cast<uint32_t>(sizeof(float) * mrgLen[i] * SORT_DATATYPE_SIZE_FACTOR), 0, 0, 0},
                {false, 0, 0, 0});
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // MrgSort
        AscendC::MrgSort4Info localParams = {mrgLen, true, VALID_QUE[queNum], 1};
        AscendC::MrgSort<float>(params.buffLocal[4],
            {params.buffLocal[0], params.buffLocal[1], params.buffLocal[2], params.buffLocal[3]},
            localParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::GetMrgSortResult(sortedLen[0], sortedLen[1], sortedLen[2], sortedLen[3]);
        const uint16_t localMrgLen = sortedLen[0] + sortedLen[1] + sortedLen[2] + sortedLen[3];
        // CopyOut
        AscendC::DataCopyPad(dstGlobal[totalMrgLen * SORT_DATATYPE_SIZE_FACTOR], params.buffLocal[4],
            {1, static_cast<uint32_t>(sizeof(float) * localMrgLen * SORT_DATATYPE_SIZE_FACTOR), 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        // Advance heads / decrement restLen
        totalMrgLen += localMrgLen;
        for (int i = 0; i < queNum; i++) {
            restLen[i] -= sortedLen[i];
            currentHead[i] += sortedLen[i];
        }
        // Compact: remove any empty queue (at most one per iteration with ifExhaustedSuspension=true)
        for (int i = 0; i < queNum; i++) {
            if (restLen[i] == 0) {
                for (int j = i; j < 3; j++) {
                    restLen[j] = restLen[j + 1];
                    currentHead[j] = currentHead[j + 1];
                    srcList[j] = srcList[j + 1];
                }
                restLen[3] = 0;
                queNum--;
                break;
            }
        }
    }
    // Tail: only 1 queue left, GM->GM copy remaining
    for (int i = 0; i < params.queNum; i++) {
        if (restLen[i] > 0) {
            DataCopyGM2GM(dstGlobal[totalMrgLen * SORT_DATATYPE_SIZE_FACTOR],
                srcList[i][currentHead[i] * SORT_DATATYPE_SIZE_FACTOR],
                params.buffLocal[4],
                restLen[i] * SORT_DATATYPE_SIZE_FACTOR, TILE_LEN_BYTE);
            break;
        }
    }
}

template <typename T>
__aicore__ inline void KernelUnique<T>::MrgBlock()
{
    if (tileNum <= 1) return;

    LocalTensor<float> sortedLocal1 = calcBuf[0].template Get<float>();
    LocalTensor<float> sortedLocal2 = calcBuf[1].template Get<float>();
    LocalTensor<float> mrgLocal = calcBuf[2].template Get<float>();
    GlobalTensor<float> sortedBlockArr[2] = {sortedBlock1, sortedBlock2};

    constexpr uint8_t PREFIX_QUE_NUM = 4;
    bool switchFlag = false;
    GlobalTensor<float> srcGM[4];
    int lengths[4];

    for (int bindTile = 1; bindTile < (int32_t)tileNum; bindTile *= PREFIX_QUE_NUM) {
        for (int tileIdx = 0; tileIdx < (int32_t)tileNum; tileIdx += bindTile * PREFIX_QUE_NUM) {
            int mrgTileNum = MIN((int32_t)tileNum - tileIdx, bindTile * PREFIX_QUE_NUM);
            uint8_t queNum = (mrgTileNum + bindTile - 1) / bindTile;
            uint8_t lastQueTileNum = mrgTileNum % bindTile;
            if (lastQueTileNum == 0) {
                lastQueTileNum = bindTile;
            }
            for (int i = 0; i < queNum; i++) {
                srcGM[i] = sortedBlockArr[switchFlag][TILE_LEN_ELEM * (tileIdx + bindTile * i)];
            }
            for (int i = 0; i < queNum - 1; i++) {
                lengths[i] = TILE_LENGTH * bindTile;
            }
            lengths[queNum - 1] = TILE_LENGTH * lastQueTileNum;

            GMSSrcList srcList{srcGM};
            GMSParams params{lengths, queNum,
                {sortedLocal1, sortedLocal1[TILE_LENGTH], sortedLocal2, sortedLocal2[TILE_LENGTH], mrgLocal}};
            MrgSortGM(sortedBlockArr[!switchFlag][TILE_LEN_ELEM * tileIdx], srcList, params);
        }
        switchFlag = !switchFlag;
    }
    // Ensure final result is in sortedBlock1
    if (switchFlag) {
        DataCopyGM2GM(sortedBlock1, sortedBlock2, sortedLocal1,
            (int)(blockLength * SORT_DATATYPE_SIZE_FACTOR), TILE_LEN_BYTE);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}


template <typename T>
__aicore__ inline void KernelUnique<T>::MrgGlobal()
{
    if (blockNum <= 1) return;

    LocalTensor<float> sortedLocal1 = calcBuf[0].template Get<float>();
    LocalTensor<float> sortedLocal2 = calcBuf[1].template Get<float>();
    LocalTensor<float> mrgLocal = calcBuf[2].template Get<float>();
    LocalTensor<int32_t> IBSyncLocal = sortedLocal2.ReinterpretCast<int32_t>();
    GlobalTensor<float> sortedGlobalArr[2] = {sortedGlobal1, sortedGlobal2};

    constexpr uint8_t PREFIX_QUE_NUM = 4;
    bool switchFlag = false;
    GlobalTensor<float> srcGM[4];
    int lengths[4];

    for (int bindBlock = 1; bindBlock < (int32_t)blockNum; bindBlock *= PREFIX_QUE_NUM, eventID++) {
        for (int blockIdx = 0; blockIdx < (int32_t)blockNum; blockIdx += bindBlock * PREFIX_QUE_NUM) {
            if ((int32_t)GetBlockIdx() == blockIdx + bindBlock ||
                (int32_t)GetBlockIdx() == blockIdx + bindBlock * 2 ||
                (int32_t)GetBlockIdx() == blockIdx + bindBlock * 3) {
                // Non-leader: signal readiness
                AscendC::PipeBarrier<PIPE_ALL>();
                IBSet(IBSyncGlobal, IBSyncLocal, (int32_t)GetBlockIdx(), eventID);
                AscendC::PipeBarrier<PIPE_ALL>();
            } else if ((int32_t)GetBlockIdx() == blockIdx) {
                // Leader: wait for non-leaders, then merge
                int mrgBlockNum = MIN((int32_t)blockNum - blockIdx, bindBlock * PREFIX_QUE_NUM);
                uint8_t queNum = (mrgBlockNum + bindBlock - 1) / bindBlock;
                for (int i = 1; i < queNum; i++) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    IBWait(IBSyncGlobal, IBSyncLocal, (int32_t)blockIdx + bindBlock * i, eventID);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                uint8_t lastQueBlockNum = mrgBlockNum % bindBlock;
                if (lastQueBlockNum == 0) {
                    lastQueBlockNum = bindBlock;
                }
                for (int i = 0; i < queNum; i++) {
                    srcGM[i] = sortedGlobalArr[switchFlag][GetGlobalOffset(blockIdx + bindBlock * i) * SORT_DATATYPE_SIZE_FACTOR];
                }
                for (int i = 0; i < queNum - 1; i++) {
                    lengths[i] = (int32_t)(GetGlobalOffset(blockIdx + bindBlock * (i + 1)) - GetGlobalOffset(blockIdx + bindBlock * i));
                }
                lengths[queNum - 1] = (int32_t)(GetGlobalOffset(blockIdx + bindBlock * (queNum - 1) + lastQueBlockNum) -
                                                GetGlobalOffset(blockIdx + bindBlock * (queNum - 1)));

                GMSSrcList srcList{srcGM};
                GMSParams params{lengths, queNum,
                    {sortedLocal1, sortedLocal1[TILE_LENGTH], sortedLocal2, sortedLocal2[TILE_LENGTH], mrgLocal}};
                MrgSortGM(sortedGlobalArr[!switchFlag][GetGlobalOffset(blockIdx) * SORT_DATATYPE_SIZE_FACTOR],
                          srcList, params);
            }
        }
        switchFlag = !switchFlag;
    }

    // Swap so final result is in sortedGlobal1 / sortedBlock1 (also swap int32 views)
    if (switchFlag) {
        GlobalTensor<float> tmpGlobal = sortedGlobal1;
        sortedGlobal1 = sortedGlobal2;
        sortedGlobal2 = tmpGlobal;

        GlobalTensor<float> tmpBlock = sortedBlock1;
        sortedBlock1 = sortedBlock2;
        sortedBlock2 = tmpBlock;

        GlobalTensor<int32_t> tmpBlockInt = sortedBlock1AsInt32;
        sortedBlock1AsInt32 = sortedBlock2AsInt32;
        sortedBlock2AsInt32 = tmpBlockInt;
    }
}



template <typename T>
__aicore__ inline void KernelUnique<T>::CalculateFlip()
{
    LocalTensor<float> tmpBuf = calcBuf[0].template Get<float>();
    uint64_t mask[1] = {0x5555555555555555ULL};
    for (uint32_t i = 0; i < tileNum; i++) {
        //计算真实长度
        int32_t remaining = (int32_t)blockRealLength - (int32_t)(i * TILE_LENGTH);
        if (remaining <= 0) break;
        int32_t validLen = MIN((int32_t)TILE_LENGTH, remaining);
        // 因为muls一次只能处理256B x 255repeat 所以分两段处理
        AscendC::DataCopyPad(tmpBuf, sortedBlock1[i * TILE_LEN_ELEM], 
            {1, static_cast<uint32_t>(TILE_LEN_ELEM * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls<float>(tmpBuf, tmpBuf, static_cast<float>(-1), mask, 128, {1, 1, 8, 8});
        AscendC::Muls<float>(tmpBuf[TILE_LENGTH], tmpBuf[TILE_LENGTH], static_cast<float>(-1), mask, 128, {1, 1, 8, 8});
        SyncDiffPipe<AscendC::HardEvent::V_MTE3>();
        // 这里写回block需要额外注意，不要把填充的-inf（经过上面的处理已经变成+inf了）也写回去了
        AscendC::DataCopyPad(sortedBlock1[i * TILE_LEN_ELEM], tmpBuf, 
            {1, static_cast<uint32_t>(validLen * 2 * sizeof(float)), 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}


template <typename T>
__aicore__ inline void KernelUnique<T>::CopyOut()
{
    CopyOutUnique();
    if (flagCounts) {
        CopyOutCounts();
        PipeBarrier<PIPE_ALL>();
    }
    if (flagInverse) {
        CopyOutInverse();
        PipeBarrier<PIPE_ALL>();
    }
}

}

#endif // UNIQUE_V3_H
