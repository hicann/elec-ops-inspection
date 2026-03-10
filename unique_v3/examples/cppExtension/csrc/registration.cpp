
#include <torch/library.h>
#include <torch/extension.h>
#include "function.h"


TORCH_LIBRARY(myops, m) {
    m.def("unique_v3(Tensor input, bool return_inverse, bool return_counts) -> (Tensor, Tensor, Tensor, Tensor)");
}

// 通过pybind将c++接口和python接口绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unique_v3", &unique_v3_custom, "unique operation on NPU");
}
