/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_all_def.cpp
 * \brief ReduceAll operator definition
 * 
 * out = ∧_axes self (逻辑与归约)
 */

#include "register/op_def_registry.h"

namespace ops {

class ReduceAll : public OpDef {
public:
    explicit ReduceAll(const char* name) : OpDef(name)
    {
        // ========== 输入定义 ==========
        
        // 输入0: self - 布尔张量
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // this->Input("axes")
        //     .ParamType(REQUIRED)
        //     .DataType({ge::DT_INT64})
        //     .Format({ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND})
        //     .AutoContiguous()
        //     .ValueDepend(REQUIRED);

        // ========== 输出定义 ==========
        
        // 输出: out - 归约结果
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // ========== 属性定义 ==========
        // 属性: dim - 指定归约的轴 (对应 aclIntArray)
        this->Attr("dim")
            .AttrType(OPTIONAL)
            .ListInt({});
        
        // 属性: keepdim - 是否保留归约后的维度
        this->Attr("keepdim")
            .AttrType(OPTIONAL)
            .Bool(false);  // 默认值为 false

        // ========== AI Core 配置 ==========
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ReduceAll);

} // namespace ops
