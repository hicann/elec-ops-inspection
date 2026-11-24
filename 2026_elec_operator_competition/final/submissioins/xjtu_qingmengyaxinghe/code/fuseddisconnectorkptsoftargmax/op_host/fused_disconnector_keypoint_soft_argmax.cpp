// Copyright (c) 2026 Qianqiuer.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
namespace optiling {
static ge::graphStatus Tile(gert::TilingContext* c) {
 const auto& x=c->GetInputShape(0)->GetStorageShape();
 const auto& w=c->GetInputShape(1)->GetStorageShape();
 const auto& b=c->GetInputShape(2)->GetStorageShape();
 if(x.GetDimNum()!=4 || w.GetDimNum()!=4 || b.GetDimNum()!=1) return ge::GRAPH_FAILED;
 if(x.GetDim(0)<1 || x.GetDim(0)>4 || x.GetDim(1)!=32 || w.GetDim(0)!=4 || w.GetDim(1)!=32 || w.GetDim(2)!=3 || w.GetDim(3)!=3 || b.GetDim(0)!=4) return ge::GRAPH_FAILED;
 const auto h=x.GetDim(2), width=x.GetDim(3);
 if(!((h==48&&width==64)||(h==64&&width==64)||(h==72&&width==96))) return ge::GRAPH_FAILED;
 const auto* tau=c->GetAttrs()->GetFloat(0); const auto* stride=c->GetAttrs()->GetInt(1);
 if(!tau || !stride || !std::isfinite(*tau) || *tau<=0 || *stride!=4) return ge::GRAPH_FAILED;
 auto* t=c->GetTilingData<KeypointTiling>();
 if(!t) return ge::GRAPH_FAILED;
 *t={static_cast<uint32_t>(x.GetDim(0)),static_cast<uint32_t>(h),static_cast<uint32_t>(width),static_cast<uint32_t>(h*width),*tau,4,1};
 platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
 const uint32_t budget=platform.GetCoreNumAiv()/(t->batch*4);
 if(budget==0) return ge::GRAPH_FAILED;
 // Whole groups occupy at most the physical AIV count; cap fragmentation at8.
 t->parts=budget<4?budget:4;
 while(t->parts*2<=budget && t->parts<8) t->parts*=2;
 const uint32_t maxRows=(t->height+t->parts-1)/t->parts;
 const uint32_t vectorElements=(maxRows+7)/8*8*t->width;
 const uint32_t maskWords=(vectorElements/32+63)/64*64;
 const uint32_t fixed=6*vectorElements+maskWords+320+2304+128;
 if((fixed+(maxRows+2)*t->width+16)*4>184*1024) return ge::GRAPH_FAILED;
 c->SetBlockDim(t->batch*4*t->parts); c->SetTilingKey(0);
 if(c->SetScheduleMode(1)!=ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

 c->GetWorkspaceSizes(1)[0]=0;
 return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus Shape(gert::InferShapeContext* c) {
 const auto* x=c->GetInputShape(0); if(!x || x->GetDimNum()!=4) return GRAPH_FAILED;
 *c->GetOutputShape(0)={x->GetDim(0),4,2}; *c->GetOutputShape(1)={x->GetDim(0),4}; return GRAPH_SUCCESS;
}
static graphStatus Type(gert::InferDataTypeContext* c) {c->SetOutputDataType(0,DT_FLOAT);c->SetOutputDataType(1,DT_FLOAT);return GRAPH_SUCCESS;}
}
namespace ops {
class FusedDisconnectorKeypointSoftArgmax : public OpDef {
public:
 explicit FusedDisconnectorKeypointSoftArgmax(const char* n):OpDef(n) {
  for(auto name:{"feat","weight","bias"}) Input(name).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
  for(auto name:{"keypoints","scores"}) Output(name).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
  Attr("tau").AttrType(REQUIRED).Float(); Attr("stride").AttrType(REQUIRED).Int();
  SetInferShape(ge::Shape).SetInferDataType(ge::Type); AICore().SetTiling(optiling::Tile).AddConfig("ascend910b");
 }
};
OP_ADD(FusedDisconnectorKeypointSoftArgmax);
}
IMPL_OP_OPTILING(FusedDisconnectorKeypointSoftArgmax).Tiling(optiling::Tile).InferShape(ge::Shape).InferDataType(ge::Type);
