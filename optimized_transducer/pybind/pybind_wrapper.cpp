// Copyright 2026 Electrical Engineering SIG - CANN Community
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

static aclDataType toAclDtype(torch::ScalarType t) {
    switch (t) {
        case torch::kFloat32: return ACL_FLOAT;
        case torch::kFloat16: return ACL_FLOAT16;
        case torch::kInt32:   return ACL_INT32;
        default: return ACL_FLOAT;
    }
}

// 原始 Tensor 的引用，确保生命周期
static aclTensor* toAclTensor(const torch::Tensor& t) {
    std::vector<int64_t> sh(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> st(t.strides().begin(), t.strides().end());
    return aclCreateTensor(sh.data(), sh.size(), toAclDtype(t.scalar_type()), st.data(), 0, 
                           ACL_FORMAT_ND, sh.data(), sh.size(), t.data_ptr());
}

static torch::Tensor npuEmpty(const std::vector<int64_t>& shape, torch::ScalarType dtype) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    size_t bytes = std::max<size_t>(n * torch::elementSize(dtype), 1);
    auto* alloc = c10_npu::NPUCachingAllocator::get();
    auto storage = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(), bytes, alloc->allocate(bytes), alloc, true);
    auto impl = c10::make_intrusive<c10::TensorImpl>(
        std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        caffe2::TypeMeta::fromScalarType(dtype));
    impl->set_sizes_contiguous(shape);
    return torch::Tensor(std::move(impl));
}

std::tuple<torch::Tensor, torch::Tensor> rnnt_loss(
    const torch::Tensor& logits, const torch::Tensor& targets,
    const torch::Tensor& logit_lens, const torch::Tensor& target_lens, int64_t blank) {
    
    // 1. 确保所有输入都是连续的，并保持对象在作用域内，防止显存提前释放
    auto logits_ = logits.contiguous();
    auto targets_ = targets.contiguous();
    auto logit_lens_ = logit_lens.contiguous();
    auto target_lens_ = target_lens.contiguous();

    int64_t B = logit_lens_.size(0);
    auto loss = npuEmpty({B}, torch::kFloat32);
    auto grad = npuEmpty(logits_.sizes().vec(), torch::kFloat32);
    
    // 2. 创建 aclTensor 描述符
    std::vector<aclTensor*> ts = {
        toAclTensor(logits_), toAclTensor(targets_),
        toAclTensor(logit_lens_), toAclTensor(target_lens_),
        toAclTensor(loss), toAclTensor(grad)
    };
    
    // 3. 获取需要的 Workspace 大小
    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnOptimizedTransducerGetWorkspaceSize(
        ts[0], ts[1], ts[2], ts[3], blank, -1.0, true, ts[4], ts[5], &wsSize, &exec);
    
    if (ret != ACL_SUCCESS) {
        for (auto* t : ts) if (t) aclDestroyTensor(t);
        throw std::runtime_error("aclnnOptimizedTransducerGetWorkspaceSize failed");
    }

    // 4. 使用 PyTorch NPU 分配器管理 Workspace，自动处理流同步释放
    torch::Tensor workspace;
    void* ws_ptr = nullptr;
    if (wsSize > 0) {
        workspace = npuEmpty({(int64_t)wsSize}, torch::kInt8);
        ws_ptr = workspace.data_ptr();
    }
    
    // 5. 执行算子
    auto stream = static_cast<aclrtStream>(c10_npu::getCurrentNPUStream(0).stream());
    ret = aclnnOptimizedTransducer(ws_ptr, wsSize, exec, stream);
    
    if (ret != ACL_SUCCESS) {
        for (auto* t : ts) if (t) aclDestroyTensor(t);
        throw std::runtime_error("aclnnOptimizedTransducer failed");
    }

    // 6. 清理描述符（不释放显存，显存由 PyTorch Tensor 管理）
    for (auto* t : ts) {
        if (t) aclDestroyTensor(t);
    }
    
    return {loss, grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rnnt_loss", &rnnt_loss, "RNN-T loss on NPU",
          py::arg("logits"), py::arg("targets"), 
          py::arg("logit_lengths"), py::arg("target_lengths"), py::arg("blank") = 0);
}