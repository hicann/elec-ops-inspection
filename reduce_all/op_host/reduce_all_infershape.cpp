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
 * \file reduce_all_infershape.cpp
 * \brief ReduceAll operator shape inference
 */

#include <vector>
#include <algorithm>
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeReduceAll(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeReduceAll");

    const gert::Shape* selfShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfShape);

    gert::Shape* outShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    auto attrs = context->GetAttrs();
    
    bool keepdim = false;
    if (attrs != nullptr) {
        const bool* keepdimPtr = attrs->GetBool(1);  // keepdim
        if (keepdimPtr != nullptr) {
            keepdim = *keepdimPtr;
        }
    }

    size_t inputRank = selfShape->GetDimNum();
    
    std::vector<int64_t> reduceAxes;
    bool isFullReduce = false;
    
    if (attrs != nullptr) {
        auto dimListPtr = attrs->GetListInt(0);  // dim
        if (dimListPtr != nullptr && dimListPtr->GetSize() > 0) {
            const int64_t* dimData = dimListPtr->GetData();
            if (dimData != nullptr) {
                for (size_t i = 0; i < dimListPtr->GetSize(); ++i) {
                    int64_t axis = dimData[i];
                    if (axis < 0) {
                        axis += static_cast<int64_t>(inputRank);
                    }
                    if (axis >= 0 && axis < static_cast<int64_t>(inputRank)) {
                        reduceAxes.push_back(axis);
                    }
                }
            }
        }
    }
    
    if (reduceAxes.empty()) {
        isFullReduce = true;
        for (size_t i = 0; i < inputRank; ++i) {
            reduceAxes.push_back(static_cast<int64_t>(i));
        }
    }

    std::sort(reduceAxes.begin(), reduceAxes.end());
    reduceAxes.erase(std::unique(reduceAxes.begin(), reduceAxes.end()), reduceAxes.end());

    if (reduceAxes.size() == inputRank) {
        isFullReduce = true;
    }

    if (keepdim) {
        outShape->SetDimNum(inputRank);
        
        std::vector<bool> isReduceAxis(inputRank, false);
        for (int64_t axis : reduceAxes) {
            isReduceAxis[static_cast<size_t>(axis)] = true;
        }
        
        for (size_t i = 0; i < inputRank; ++i) {
            if (isReduceAxis[i]) {
                outShape->SetDim(i, 1);  // 规约轴变为1
            } else {
                outShape->SetDim(i, selfShape->GetDim(i));  // 非规约轴保持不变
            }
        }
    } else {
        if (isFullReduce) {
            // full reduce + keepdim=false => [1]
            outShape->SetDimNum(1);
            outShape->SetDim(0, 1);
        } else {
            std::vector<bool> isReduceAxis(inputRank, false);
            for (int64_t axis : reduceAxes) {
                isReduceAxis[static_cast<size_t>(axis)] = true;
            }
            
            size_t outputRank = 0;
            for (size_t i = 0; i < inputRank; ++i) {
                if (!isReduceAxis[i]) {
                    outputRank++;
                }
            }
            
            outShape->SetDimNum(outputRank);
            
            size_t outIdx = 0;
            for (size_t i = 0; i < inputRank; ++i) {
                if (!isReduceAxis[i]) {
                    outShape->SetDim(outIdx, selfShape->GetDim(i));
                    outIdx++;
                }
            }
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeReduceAll");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReduceAll).InferShape(InferShapeReduceAll);

} // namespace ops