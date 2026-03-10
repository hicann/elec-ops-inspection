#ifndef FUNCTION_H
#define FUNCTION_H

#include <ATen/ATen.h>

// Unique_v3 算子接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> unique_v3_custom(
    const at::Tensor& input, 
    const bool return_inverse = false, 
    const bool return_counts = false);

#endif //  FUNCTION_H
