/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file unique_v3_custom.cpp
 *
 * Unique_v3 算子的 PyTorch NPU 适配层
 */
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <cstdio>

// 为 NPU 设备注册前向实现
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> unique_v3_impl_npu(
    const at::Tensor& input,
    const bool flag_inverse,
    const bool flag_counts) {
    at::Tensor output = at::empty_like(input);
    at::Tensor uniqueCnt = at::empty({1}, input.options().dtype(at::kInt));

    //at::Tensor inverse = at::empty_like(input, input.options().dtype(at::kInt));
    //at::Tensor counts = at::empty_like(input, input.options().dtype(at::kInt));

    // 计算对齐后的大小
    int64_t numel = input.numel();
    int64_t aligned_size = (numel + 8191) / 8192 * 8192;
    // 使用对齐后的大小创建 tensor
    at::Tensor inverse = at::empty({aligned_size}, input.options().dtype(at::kInt));
    at::Tensor counts = at::empty({aligned_size}, input.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnUniqueV3, input, flag_inverse, flag_counts, output, uniqueCnt, inverse, counts);
    return std::make_tuple(output, uniqueCnt, inverse, counts);
}


// 为 NPU 设备注册实现
// NPU 设备在 pytorch 2.1 及以上版本使用的设备名称是 PrivateUse1
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("unique_v3", &unique_v3_impl_npu);
}


// 包装函数，供 Python 调用
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> unique_v3_custom(
    const at::Tensor& input,
    const bool return_inverse,
    const bool return_counts) {
    static auto op = torch::Dispatcher::singleton()
                    .findSchemaOrThrow("myops::unique_v3", "")
                    .typed<decltype(unique_v3_impl_npu)>();
    return op.call(input, return_inverse, return_counts);
}
