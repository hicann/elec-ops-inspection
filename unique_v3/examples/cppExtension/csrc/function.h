/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUNCTION_H
#define FUNCTION_H

#include <ATen/ATen.h>

// Unique_v3 算子接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> unique_v3_custom(
    const at::Tensor& input, 
    const bool return_inverse = false, 
    const bool return_counts = false);

#endif //  FUNCTION_H
