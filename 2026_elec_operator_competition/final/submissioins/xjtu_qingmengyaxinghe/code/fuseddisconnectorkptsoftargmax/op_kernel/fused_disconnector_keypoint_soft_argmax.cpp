// Copyright (c) 2026 Qianqiuer.
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "kernel_common.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
// Copyright (c) 2026 Qianqiuer.
// SPDX-License-Identifier: Apache-2.0
namespace AivSpatial {
using namespace AscendC;
class Core {
public:
 __aicore__ inline void Init(GM_ADDR f,GM_ADDR w,GM_ADDR b,const KeypointTiling& t) {
  width=t.width; totalHeight=t.height; spatial=t.spatial; tau=t.tau;
  group=GetBlockIdx()/t.parts; part=GetBlockIdx()%t.parts;
  start=totalHeight*part/t.parts; rows=totalHeight*(part+1)/t.parts-start;
  n=rows*width; aligned=((rows+7)/8*8)*width;
  maskWords=(aligned/32+63)/64*64;
  channelStride=(rows+2)*width+16;
  const uint32_t bufferStride=aligned+8; // 32B phase shift between vector operands.
  const uint32_t fixed=6*bufferStride+maskWords+320+2304+128;
  channels=32;
  while(channels>1 && (fixed+channels*channelStride)*4>184*1024) channels/=2;
  pipe.InitBuffer(arena,(fixed+channels*channelStride)*4);
  auto base=arena.Get<float>();
  heat=base; compensation=base[bufferStride]; dot=base[2*bufferStride]; gathered=base[3*bufferStride];
  temporary=base[4*bufferStride]; indices=base[5*bufferStride].ReinterpretCast<uint32_t>();
  mask=base[6*bufferStride]; coefficients=base[6*bufferStride+maskWords];
  broadcast=base[6*bufferStride+maskWords+320];
  record=base[6*bufferStride+maskWords+320+2304]; merge=record[64]; cache=base[fixed];
  feat.SetGlobalBuffer((__gm__ float*)f+(group/4)*32*spatial);
  weight.SetGlobalBuffer((__gm__ float*)w+(group%4)*288);
  bias.SetGlobalBuffer((__gm__ float*)b);
 }
 __aicore__ inline void Convolve() {
  DataCopy(coefficients,weight,288);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Brcb(broadcast,coefficients,36,BrcbRepeatParams{1,8}); PipeBarrier<PIPE_V>();
  Duplicate(heat,0.0F,aligned); Duplicate(compensation,0.0F,aligned); PipeBarrier<PIPE_V>();
  if(channels==32) LoadChannels(0);
  for(uint32_t tap=0;tap<9;++tap) {
   MakeTapIndices(tap);
   Duplicate(dot,0.0F,n); PipeBarrier<PIPE_V>();
   for(uint32_t channelBase=0;channelBase<32;channelBase+=channels) {
    if(channels!=32) LoadChannels(channelBase);
    for(uint32_t channel=0;channel<channels;++channel) {
     Gather(gathered,cache[channel*channelStride],indices,0,n); PipeBarrier<PIPE_V>();
     if(tap%3!=1) {
      Select(gathered,mask.ReinterpretCast<uint8_t>(),gathered,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,n);
      PipeBarrier<PIPE_V>();
     }
     auto coefficient=broadcast[((channelBase+channel)*9+tap)*8];
     MulAddDst(dot,gathered,coefficient,uint64_t{64},static_cast<uint8_t>(n/64),BinaryRepeatParams{1,1,0,8,8,0});
     if(n%64) MulAddDst(dot[n/64*64],gathered[n/64*64],coefficient,static_cast<uint64_t>(n%64),1,BinaryRepeatParams{1,1,0,8,8,0});
     PipeBarrier<PIPE_V>();
    }
   }
   Sub(dot,dot,compensation,n); PipeBarrier<PIPE_V>();
   Add(temporary,heat,dot,n); PipeBarrier<PIPE_V>();
   Sub(compensation,temporary,heat,n); PipeBarrier<PIPE_V>();
   Sub(compensation,compensation,dot,n); PipeBarrier<PIPE_V>();
   auto previous=heat; heat=temporary; temporary=previous;
  }
  Adds(heat,heat,bias.GetValue(group%4),n); PipeBarrier<PIPE_V>();
 }
 __aicore__ inline float LocalMaximum() {
  // Include only true spatial values in ReduceMax, but detect NaN explicitly.
  // All padding in heat/temporary is initialized before their Kahan role swaps.
  Duplicate(mask.ReinterpretCast<int32_t>(),int32_t{-1},maskWords); PipeBarrier<PIPE_V>();
  Compare(mask.ReinterpretCast<uint8_t>(),heat,heat,CMPMODE::EQ,aligned); PipeBarrier<PIPE_V>();
  Cast(dot,mask.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,maskWords); PipeBarrier<PIPE_V>();
  Adds(dot,dot,1.0F,maskWords); PipeBarrier<PIPE_V>(); Abs(dot,dot,maskWords); PipeBarrier<PIPE_V>();
  ReduceMax(record,dot,gathered,maskWords,false);
  SetFlag<HardEvent::V_S>(EVENT_ID6); WaitFlag<HardEvent::V_S>(EVENT_ID6);
  const bool hasNaN=record.GetValue(0)!=0.0F;
  SetFlag<HardEvent::S_V>(EVENT_ID6); WaitFlag<HardEvent::S_V>(EVENT_ID6);
  if(hasNaN) { union {uint32_t bits;float value;} qnan={0x7fc00000U}; return qnan.value; }
  ReduceMax(record,heat,gathered,n,false);
  SetFlag<HardEvent::V_S>(EVENT_ID6); WaitFlag<HardEvent::V_S>(EVENT_ID6);
  const float maximum=record.GetValue(0);
  SetFlag<HardEvent::S_V>(EVENT_ID6); WaitFlag<HardEvent::S_V>(EVENT_ID6);
  return maximum;
 }
 __aicore__ inline LocalTensor<float> WeightedSums(float maximum) {
  Adds(heat,heat,-maximum,n); PipeBarrier<PIPE_V>();
  Duplicate(dot,tau,n); PipeBarrier<PIPE_V>(); Div(heat,heat,dot,n); PipeBarrier<PIPE_V>();
  Exp(heat,heat,n); PipeBarrier<PIPE_V>();
  ReduceSum(record,heat,gathered,n); PipeBarrier<PIPE_V>();
  MakeX(temporary);
  Mul(dot,temporary,heat,n); PipeBarrier<PIPE_V>();
  ReduceSum(record[8],dot,gathered,n); PipeBarrier<PIPE_V>();
  CreateVecIndex(temporary,static_cast<float>(static_cast<int32_t>(start)),(rows+7)/8*8); PipeBarrier<PIPE_V>();
  Brcb(dot,temporary,static_cast<uint8_t>((rows+7)/8),BrcbRepeatParams{static_cast<uint16_t>(width/8),static_cast<uint16_t>(width)}); PipeBarrier<PIPE_V>();
  Copy(dot[8],dot,uint64_t{56},static_cast<uint8_t>(rows),CopyRepeatParams{1,0,static_cast<uint16_t>(width/8),static_cast<uint16_t>(width/8)});
  if(width>64) Copy(dot[64],dot,static_cast<uint64_t>(width-64),static_cast<uint8_t>(rows),CopyRepeatParams{1,0,static_cast<uint16_t>(width/8),static_cast<uint16_t>(width/8)});
  PipeBarrier<PIPE_V>(); Mul(dot,dot,heat,n); PipeBarrier<PIPE_V>();
  ReduceSum(record[16],dot,gathered,n);
  SetFlag<HardEvent::V_S>(EVENT_ID6); WaitFlag<HardEvent::V_S>(EVENT_ID6);
  return record;
 }
 __aicore__ inline LocalTensor<float> MergeScratch() {return merge;}
 __aicore__ inline LocalTensor<float> Heat() {return heat;}
 __aicore__ inline uint32_t Group() const {return group;}
 __aicore__ inline uint32_t Part() const {return part;}
 __aicore__ inline uint32_t StartRow() const {return start;}
 __aicore__ inline uint32_t Count() const {return n;}
private:
 __aicore__ inline void MakeX(LocalTensor<float> target) {
  Duplicate(target,0.0F,aligned); PipeBarrier<PIPE_V>();
  CreateVecIndex(target,0.0F,width); PipeBarrier<PIPE_V>();
  if(rows>1) {
   Copy(target[width],target,uint64_t{64},static_cast<uint8_t>(rows-1),CopyRepeatParams{1,1,static_cast<uint16_t>(width/8),0});
   if(width>64) Copy(target[width+64],target[64],static_cast<uint64_t>(width-64),static_cast<uint8_t>(rows-1),CopyRepeatParams{1,1,static_cast<uint16_t>(width/8),0});
  }
  PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void MakeTapIndices(uint32_t tap) {
  const int32_t dy=static_cast<int32_t>(tap/3)-1,dx=static_cast<int32_t>(tap%3)-1;
  CreateVecIndex(temporary,0.0F,n); PipeBarrier<PIPE_V>();
  Muls(temporary,temporary,4.0F,n); PipeBarrier<PIPE_V>();
  Adds(temporary,temporary,static_cast<float>((8+(dy+1)*static_cast<int32_t>(width)+dx)*4),n); PipeBarrier<PIPE_V>();
  Cast(indices.ReinterpretCast<int32_t>(),temporary,RoundMode::CAST_RINT,n); PipeBarrier<PIPE_V>();
  if(dx!=0) {
   MakeX(temporary);
   CompareScalar(mask.ReinterpretCast<uint8_t>(),temporary,dx<0?0.0F:static_cast<float>(static_cast<int32_t>(width-1)),CMPMODE::NE,aligned);
   PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void LoadChannels(uint32_t channelBase) {
  Duplicate(cache,0.0F,channels*channelStride);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
  const uint32_t top=start==0?1:0,first=start==0?0:start-1;
  const uint32_t bottom=start+rows==totalHeight?1:0,validRows=rows+2-top-bottom;
  const DataCopyExtParams params{static_cast<uint16_t>(channels),validRows*width*4,
   (spatial-validRows*width)*4,(channelStride-validRows*width)/8,0};
  const DataCopyPadExtParams<float> padding{false,0,0,0.0F};
  DataCopyPad(cache[8+top*width],feat[channelBase*spatial+first*width],params,padding);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
 }
 uint32_t width,totalHeight,spatial,group,part,start,rows,n,aligned,maskWords,channelStride,channels;
 float tau;
 TPipe pipe; TBuf<TPosition::VECCALC> arena;
 GlobalTensor<float> feat,weight,bias;
 LocalTensor<float> heat,compensation,dot,gathered,temporary,mask,coefficients,broadcast,record,merge,cache;
 LocalTensor<uint32_t> indices;
};
}

// Copyright (c) 2026 Qianqiuer.
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

// Requires KERNEL_TYPE_MIX_AIV_1_0, SetScheduleMode(1), and logical
// numBlocks <= physical AIV count. Every launched AIV must call both phases.
// No TPipe is created here: the caller owns a live TPipe and a private,
// 32-byte-aligned scratch allocation of at least 64 floats.
namespace AivOutputScratch {
using namespace AscendC;
class Merge {
public:
 __aicore__ inline void Init(GM_ADDR keys,GM_ADDR scores,uint32_t group,
                            uint32_t part,const LocalTensor<float>& scratch) {
  leader=part==0; work=scratch;
  keyGm.SetGlobalBuffer((__gm__ float*)keys+group*2);
  scoreGm.SetGlobalBuffer((__gm__ float*)scores+group);
  scoreBits.SetGlobalBuffer((__gm__ int32_t*)scores+group);
 }
 __aicore__ inline float GlobalMaximum(float localMaximum) {
  // A signed integer ordering preserves every finite/infinite FP32 value;
  // canonical NaN dominates all numbers, independent of FP32 atomic-max NaN rules.
  union Word { float f; uint32_t u; } value;
  value.f=localMaximum;
  const uint32_t magnitude=value.u&0x7fffffffU;
  const uint32_t ordered=magnitude>0x7f800000U?0x7fffffffU:
      ((value.u&0x80000000U)?(value.u^0x7fffffffU):value.u);
  auto bits=work.ReinterpretCast<int32_t>();
  bits.SetValue(0,static_cast<int32_t>(ordered));
  ScalarToMte3();
  // Initialize every destination before any atomic read-modify-write.
  if(leader) DataCopyPad(scoreBits,bits,DataCopyExtParams{1,4,0,0,0});
  SyncAll<true>(); // 1: initial score bits are globally available.
  SetAtomicMax<int32_t>();
  DataCopyPad(scoreBits,bits,DataCopyExtParams{1,4,0,0,0});
  // Drain MTE3 before changing its global atomic mode.
  SetFlag<HardEvent::MTE3_S>(EVENT_ID6); WaitFlag<HardEvent::MTE3_S>(EVENT_ID6);
  SetAtomicNone();
  SyncAll<true>(); // 2: all partitions have contributed their maximum.
  DataCopyPad(bits[8],scoreBits,DataCopyExtParams{1,4,0,0,0},
              DataCopyPadExtParams<int32_t>{false,0,0,0});
  Mte2ToScalar();
  const uint32_t encoded=static_cast<uint32_t>(bits.GetValue(8));
  value.u=encoded==0x7fffffffU?0x7fc00000U:
      ((encoded&0x80000000U)?(encoded^0x7fffffffU):encoded);
  const float maximum=value.f;
  SyncAll<true>(); // 3: every AIV has saved max before score becomes denominator.
  if(leader) {
   work.SetValue(16,0.0F); work.SetValue(17,0.0F);
   ScalarToMte3();
   DataCopyPad(keyGm,work[16],DataCopyExtParams{1,8,0,0,0});
   DataCopyPad(scoreGm,work[16],DataCopyExtParams{1,4,0,0,0});
  }
  SyncAll<true>(); // 4: all three accumulators are initialized, not allocator contents.
  return maximum;
 }
 __aicore__ inline void Commit(float denominator,float xSum,float ySum,
                              float maximum,uint32_t stride) {
  // Each UB DMA source is 32-byte aligned; each GM destination has exact 4B extent.
  work.SetValue(0,denominator); work.SetValue(8,xSum); work.SetValue(16,ySum);
  ScalarToMte3();
  SetAtomicAdd<float>();
  DataCopyPad(scoreGm,work,DataCopyExtParams{1,4,0,0,0});
  DataCopyPad(keyGm,work[8],DataCopyExtParams{1,4,0,0,0});
  DataCopyPad(keyGm[1],work[16],DataCopyExtParams{1,4,0,0,0});
  // Drain MTE3 before changing its global atomic mode.
  SetFlag<HardEvent::MTE3_S>(EVENT_ID6); WaitFlag<HardEvent::MTE3_S>(EVENT_ID6);
  SetAtomicNone();
  SyncAll<true>(); // 5: globally complete sums; nonleaders no longer touch outputs.
  if(!leader) return;
  DataCopyPad(work,keyGm,DataCopyExtParams{1,8,0,0,0},
              DataCopyPadExtParams<float>{false,0,0,0.0F});
  DataCopyPad(work[8],scoreGm,DataCopyExtParams{1,4,0,0,0},
              DataCopyPadExtParams<float>{false,0,0,0.0F});
  Mte2ToScalar();
  const float total=work.GetValue(8);
  work.SetValue(24,maximum);
  SetFlag<HardEvent::S_V>(EVENT_ID6); WaitFlag<HardEvent::S_V>(EVENT_ID6);
  Duplicate(work[16],total,8); PipeBarrier<PIPE_V>();
  Div(work,work,work[16],2); PipeBarrier<PIPE_V>();
  Adds(work,work,0.5F,2); PipeBarrier<PIPE_V>();
  Muls(work,work,static_cast<float>(static_cast<int32_t>(stride)),2);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID6); WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
  // work[24] was written on S; establish its independent DMA dependency too.
  ScalarToMte3();
  DataCopyPad(keyGm,work,DataCopyExtParams{1,8,0,0,0});
  DataCopyPad(scoreGm,work[24],DataCopyExtParams{1,4,0,0,0});
  SetFlag<HardEvent::MTE3_S>(EVENT_ID6); WaitFlag<HardEvent::MTE3_S>(EVENT_ID6);
 }
private:
 __aicore__ inline void ScalarToMte3() {
  SetFlag<HardEvent::S_MTE3>(EVENT_ID6); WaitFlag<HardEvent::S_MTE3>(EVENT_ID6);
 }
 __aicore__ inline void Mte2ToScalar() {
  SetFlag<HardEvent::MTE2_S>(EVENT_ID6); WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
 }
 bool leader;
 LocalTensor<float> work;
 GlobalTensor<float> keyGm,scoreGm;
 GlobalTensor<int32_t> scoreBits;
};
}

using namespace AscendC;
extern "C" __global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
 GM_ADDR feat,GM_ADDR weight,GM_ADDR bias,GM_ADDR keypoints,GM_ADDR scores,GM_ADDR workspace,GM_ADDR tiling) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
 REGISTER_TILING_DEFAULT(KeypointTiling);
 GET_TILING_DATA_WITH_STRUCT(KeypointTiling,t,tiling);
 AivSpatial::Core core;
 core.Init(feat,weight,bias,t); core.Convolve();
 const float localMaximum=core.LocalMaximum();
 AivOutputScratch::Merge merge;
 merge.Init(keypoints,scores,core.Group(),core.Part(),core.MergeScratch());
 const float maximum=merge.GlobalMaximum(localMaximum);
 auto stats=core.WeightedSums(maximum);
 merge.Commit(stats.GetValue(0),stats.GetValue(8),stats.GetValue(16),maximum,t.stride);
}
