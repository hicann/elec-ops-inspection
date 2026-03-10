import torch
import torch_npu
import numpy as np
import time

import custom_ops

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


def generate_test_data(seed, size=100000):
    """生成测试数据"""
    np.random.seed(seed)
    values = []
    current_val = 0
    while len(values) < size:
        repeat_count = np.random.randint(1, 20)  # 每个值重复1-20次
        values.extend([current_val] * min(repeat_count, size - len(values)))
        current_val += np.random.randint(1, 10)  # 值之间间隔1-10
    values = values[:size]
    np.random.shuffle(values)  # 打乱顺序，使数据变成乱序
    return torch.tensor(values, dtype=torch.float32)

def run_tests():
    """测试 float 类型的去重 - 30000个元素，测试多tile场景"""
    
    # 生成两组不同的测试数据
    # A 组用于预热，B 组用于正式测试
    print("Generating test data...")
    x_warmup_cpu = generate_test_data(seed=42, size=100000)  # A 组：预热数据
    x_test_cpu = generate_test_data(seed=12345, size=100000)  # B 组：测试数据
    
    x_warmup_npu = x_warmup_cpu.npu()
    x_test_npu = x_test_cpu.npu()

    print(f"Warmup data (A) size: {x_warmup_cpu.shape[0]}")
    print(f"Test data (B) size: {x_test_cpu.shape[0]}")
    print(f"Test data (B) sample (first 20): {x_test_cpu[:20].tolist()}")
    print(f"Test data (B) sample (last 20): {x_test_cpu[-20:].tolist()}")

    
    #"""
    # 预热 NPU：用 A 组数据执行 40 次
    print("\n=== Warming up NPU with data A (40 iterations) ===")
    for i in range(40):
        _ = custom_ops.unique_v3(x_warmup_npu, True, True)
    torch.npu.synchronize()
    print("Warmup done!")
    #"""

    # 在 NPU 上用 B 组数据执行 unique，测量时间
    torch.npu.synchronize()
    npu_start = time.time()
    output, unique_cnt, inverse, counts = custom_ops.unique_v3(x_test_npu, True, True)
    torch.npu.synchronize()
    npu_end = time.time()
    npu_time_ms = (npu_end - npu_start) * 1000

    # 在 CPU 上用 B 组数据执行 torch.unique，测量时间
    cpu_start = time.time()

    cpu_output, cpu_inverse, cpu_counts = torch.unique(x_test_cpu, sorted=True, return_inverse=True, return_counts=True)
    # cpu_output, cpu_counts = torch.unique(x_test_cpu, sorted=True, return_inverse=True, return_counts=True)

    cpu_end = time.time()
    cpu_time_ms = (cpu_end - cpu_start) * 1000

    # 只比较有效部分（前 unique_cnt 个元素）
    cnt = unique_cnt.item()
    print(f"\n=== Results ===")
    print(f"NPU unique count: {cnt}")
    print(f"CPU unique count: {len(cpu_output)}")
    
    print(f"\n=== Performance ===")
    print(f"NPU time: {npu_time_ms:.3f} ms")
    print(f"CPU time: {cpu_time_ms:.3f} ms")
    if npu_time_ms > 0:
        print(f"Speedup: {cpu_time_ms / npu_time_ms:.2f}x")
    
    # 比较唯一值
    npu_result = output[:cnt].cpu()
    print(f"\nUnique values match: {torch.allclose(npu_result, cpu_output)}")
    
    if cnt <= 50:
        print(f"NPU output: {npu_result.tolist()}")
        print(f"CPU output: {cpu_output.tolist()}")
    else:
        print(f"NPU output (first 20): {npu_result[:20].tolist()}")
        print(f"NPU output (last 20): {npu_result[-20:].tolist()}")
        print(f"CPU output (first 20): {cpu_output[:20].tolist()}")
        print(f"CPU output (last 20): {cpu_output[-20:].tolist()}")
    
    # 比较counts（如果实现了的话）
    if counts is not None:
        npu_counts = counts[:cnt].cpu()
        print(f"\nCounts match: {torch.equal(npu_counts, cpu_counts)}")
        if cnt <= 50:
            print(f"NPU counts: {npu_counts.tolist()}")
            print(f"CPU counts: {cpu_counts.tolist()}")
        else:
            print(f"NPU counts (first 20): {npu_counts[:20].tolist()}")
            print(f"NPU counts (last 20): {npu_counts[-20:].tolist()}")
            print(f"CPU counts (first 20): {cpu_counts[:20].tolist()}")
            print(f"CPU counts (last 20): {cpu_counts[-20:].tolist()}")
            
            # 找出不匹配的位置
            if not torch.equal(npu_counts, cpu_counts):
                diff_mask = npu_counts != cpu_counts
                diff_indices = torch.where(diff_mask)[0]
                print(f"\nMismatched indices: {diff_indices[:20].tolist()}")
                for idx in diff_indices[:5]:
                    print(f"  Index {idx}: NPU={npu_counts[idx].item()}, CPU={cpu_counts[idx].item()}")

    if inverse is not None:
        npu_inverse = inverse[:100000].cpu()
        print(f" ")
        print(f"inverse match: {torch.equal(npu_inverse, cpu_inverse)}")
        print(f"NPU inverse: {inverse[:20].tolist()}")
        print(f"CPU inverse: {cpu_inverse[:20].tolist()}")



if __name__ == "__main__":
    print("=== Large Test (30000 elements) ===")
    run_tests()
