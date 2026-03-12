# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torchaudio
import torch_npu
import time
import warnings
warnings.filterwarnings("ignore")
import optimized_transducer_ascend_ops as ops

torch_npu.npu.set_device(0)

def run_test(B, T, U, V, warmup=False):
    if not warmup:
        print(f"测试  B={B}, T={T}, U={U}, V={V}")
    torch.manual_seed(42)
    
    # 构造输入
    logits_4d = torch.randn(B, T, U, V, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(1, V, (B, U - 1), dtype=torch.int32)
    logit_lens = torch.full((B,), T, dtype=torch.int32)
    target_lens = torch.full((B,), U - 1, dtype=torch.int32)
    logits_packed = logits_4d.view(B * T * U, V)

    

    # NPU
    logits_npu = logits_packed.to("npu:0")
    targets_npu = targets.to("npu:0")
    logit_lens_npu = logit_lens.to("npu:0")
    target_lens_npu = target_lens.to("npu:0")
    torch_npu.npu.synchronize()

    # 计时
    start = time.perf_counter()
    loss_npu, grad_npu = ops.rnnt_loss(
        logits_npu, targets_npu, logit_lens_npu, target_lens_npu, 0
    )
    torch_npu.npu.synchronize()
    npu_time = time.perf_counter() - start

    if warmup:
        return


    print("=" * 60)
    print("性能")
    print(f"  NPU 耗时 : {npu_time*1000:.2f} ms")
    print("-" * 60)
    print("=" * 60 + "\n ")


# 预热
print(">>> Warmup ...")
run_test(300, 10, 10, 10, warmup=True)

# 测试
print(">>> Testing ...")

# run_test(1, 10000, 5000, 10)
run_test(1, 10, 10, 10)
# run_test(2, 10, 10, 10)
# run_test(40, 10, 10, 10)
# run_test(41, 10, 10, 10)
run_test(1, 8, 8, 2000000)