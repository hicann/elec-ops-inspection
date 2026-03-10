# rnnt-loss-ascend

**项目简介**

本项目为基于华为 CANN 计算框架开发的 RNN-T Loss 专用算子，功能上完全对齐 `torchaudio.functional.rnnt_loss`（torchaudio v2.8.0），并在此基础上引入 [optimized_transducer](https://arxiv.org/abs/1909.12415) 的内存与计算优化策略，在昇腾 NPU 上实现高效的 RNN-T 损失前向与反向联合计算。该算子支持 float32 与 float16 数据类型，可无缝嵌入语音识别训练流水线，显著降低显存占用并提升训练吞吐量。

---

**主要功能**

- **接口对齐**：完整实现 `torchaudio.functional.rnnt_loss` 的全部参数语义，包括 `blank`、`clamp`、`reduction`、`fused_log_softmax`，可直接替换现有 torchaudio 调用。
- **内存优化**：将原始 `(N, T, U, V)` 四维广播张量压缩为 `(Σ(t,u), V)` 二维紧凑表示，消除冗余内存分配。
- **梯度融合**：将 softmax 与梯度计算原地合并，省去中间大张量存储，梯度结果直接原地覆写 logits 输出缓冲区。
- **多核负载均衡**：Host 侧采用最大任务优先 + 最小负载优先策略，将批次样本均衡分配至多个 AI Core，充分利用 NPU 算力。
- **对角线并行遍历**：Kernel 侧按反对角线顺序计算 α/β 表，最大化核内数据复用，减少 HBM 访问次数。
- **双精度支持**：支持 float32 与 float16，满足不同精度与性能需求。

---

**应用场景**

| 应用领域 | 典型场景 | 说明 |
|---|---|---|
| 语音识别 | 流式 ASR 模型训练 | 以 RNN-T 为解码器的端到端语音识别，online streaming 场景 |
| 语音翻译 | 端到端语音翻译 | 多语言序列转换任务中的 transducer loss 计算 |
| 大规模预训练 | 工业级语音数据训练 | 万小时级数据下 batch size 受限问题，通过内存优化扩大批量 |
| 模型蒸馏 | 轻量化 RNN-T 模型 | 教师-学生框架中高效计算 soft target 的 transducer loss |

---

**昇腾原生 × 电力行业赋能**

**场景：电力设备巡检 Agent 交互（elec-ops-inspection）**

**业务背景**：
在特高压换流站（如主变压器室、冷却塔区域）的巡检作业现场，常年伴随 80-90 分贝的严重机械低频噪音。巡检工人作业时通常携带各类现场作业工具，在采集数据或者填写表单时，传统的平板触控或按键终端容易误触且效率低下。通过端侧算力部署离线语音 Agent 助理实现“解放双手”是未来智能巡检的发展方向，但这要求模型在强噪下具备极高的鲁棒性，且受限于边缘算力，模型训练与推理的显存调度必须极致优化。

- **算子价值**：将复杂的 Transducer Loss 内存占用降低约 50%，前反向极致融合，使得在同等算力集群下能吞吐更大 Batch Size 的长音频训练数据。
- **对应模型**：Conformer / Emformer 强抗噪流式 ASR 模型、Qwen-Audio 等端侧多模态 Agent 基座大模型。
- **应用落地**：
  1. **强噪环境离线语音指令响应（应用：巡检缺陷无接触一键录入）**：利用本算子高效的内存复用机制，可在国产算力上训练更高精度的 Conformer 工业抗噪模型。一线人员在嘈杂的变压器旁语音交互“记录二号主变油温过高并生成紧急缺陷工单”，端侧 Agent 能够以极低延迟精准转写文本，并联动内部系统接口自动派单，实现全程无接触作业。

---

**价值说明**

**为什么需要 RNN-T Loss 专用融合算子？**

**1. 内存搬运优化**

传统 RNN-T Loss 实现在训练时需维护 `(B, T, U, V)` 量级的多个大张量，以标准 softmax + 梯度分步计算为例，典型 HBM 搬运路径如下：

- 读取 logits → 计算 softmax → 写回概率张量 P（2 次搬运）
- 读取 P → 计算 ∂P/∂h → 写回雅可比张量（2 次搬运）
- 读取 P、α、β → 计算 ∂L/∂h → 写回梯度张量（6 次搬运）
- 读取梯度 → 做 reduction → 写回最终结果（2 次搬运）

总计约 **12 次以上** HBM 搬运，中间结果全部占据 HBM。

本算子融合后的优化路径：一次读取 logits 到片上 UB，在 UB 内完成 log_softmax、α/β 递推、梯度公式计算全流程，一次写回 loss 与 grad 最终结果，**总计 2 次 HBM 搬运**。

| 优化点 | 传统方案 | 融合算子 |
|---|---|---|
| HBM 搬运次数 | 12+ 次 | 2 次 |
| 中间大张量数量 | 3 个（P、∂P/∂h、∂L/∂h） | 0 个（全部原地覆写） |
| logits 内存布局 | 4D `(B,T,U,V)` 广播展开 | 2D `(Σ(t,u), V)` 紧凑排列 |
| Kernel 启动次数 | 多次（softmax、forward、backward 分离） | 1 次（全流程融合） |

**2. 计算优化**

| 优化技术 | 说明 |
|---|---|
| 2D 张量压缩 | 每条样本的 logits 按实际有效 (t,u) 位置拼接为 2D，避免 padding 位置的无效计算 |
| 对角线遍历 α/β | 按反对角线顺序遍历格子，同一对角线上各点依赖关系已满足，可批量并发处理 |
| 梯度公式化简 | 将 softmax → P → ∂L/∂h 三步融合为公式 13（见论文），直接计算梯度，省去中间变量 |
| tile 自适应分块 | 根据词表大小 V 和 UB 容量自动确定 tile 面积 M，充分利用片上缓存 |
| double buffer | 数据搬运与计算流水并发，隐藏 HBM 访问延迟 |

核心梯度融合公式如下：

$$\frac{\partial L}{\partial h_{t,u}^k} = P(k|t,u) \cdot \frac{\alpha(t,u)}{P(\mathbf{y}|\mathbf{x})} \cdot \left[\beta(t,u) - \beta(\text{next})\right]$$

其中 $P(k|t,u) = \text{softmax}(h_{t,u})_k$，$\alpha$、$\beta$ 为前向后向概率，临时变量 $P(k|t,u)$ 直接复用梯度输出缓冲区，避免额外分配。

**3. 精度保证**

| 特性 | 说明 |
|---|---|
| 双 loss 互校验 | 前向计算结束后，由 α 路径和 β 路径分别独立得到 loss，两者必须在阈值内吻合，不一致时报错 |
| 数值稳定 | 全程在 log 域进行 α/β 递推（log-sum-exp），避免概率连乘下溢 |
| clamp 保护 | 支持对梯度做 `[-clamp, clamp]` 截断，防止极端梯度破坏训练稳定性 |
| 可复现性 | 相同输入保证相同输出，满足科研可复现要求 |

---

**参数说明**

| 参数名 | 输入/输出 | 描述 | 数据类型 | 形状 |
|---|---|---|---|---|
| logits | 输入 | joiner 输出的未归一化 logits，按样本拼接为 2D | float32 / float16 | `(Σ(t_i × u_i), V)` |
| targets | 输入 | 目标序列 token，zero padding 对齐 | int32 | `(B, max_U - 1)` |
| logit_lengths | 输入 | 每条样本 encoder 输出的有效帧长 | int32 | `(B,)` |
| target_lengths | 输入 | 每条样本目标序列的真实长度 | int32 | `(B,)` |
| blank | 属性 | blank 标签 ID，-1 表示使用词表最后一个 class | int64 | — |
| clamp | 属性 | 梯度截断值，-1 表示不启用 | float32 | — |
| fused_log_softmax | 属性 | 是否在 kernel 内执行 log_softmax；若外部已做则设为 False | bool | — |
| loss | 输出 | 每条样本的 RNN-T loss | float32 / float16 | `(B,)` |
| grad | 输出 | 对 logits 的梯度，与输入 logits 同形 | float32 / float16 | `(Σ(t_i × u_i), V)` |

---

**约束说明**

- 输入 logits 必须为 float32 或 float16 类型，targets / logit_lengths / target_lengths 必须为 int32 类型。
- blank 取值范围为 `[-V, V-1]`，传入 -1 时自动映射为 `V - 1`。
- 每条样本的有效帧数 T 和目标长度 U 满足 `T ≥ 1`，`U ≥ 1`，且 `logit_lengths[i] * (target_lengths[i] + 1)` 之和等于 logits 的第 0 维大小。
- 批次大小 B 不超过可用 AI Core 数量的整数倍（超出部分由负载均衡策略处理）。
- clamp ≤ 0 时视为不启用梯度截断。

---

**架构设计**

```
Python 层（autograd Function）
    │  rnnt_loss() → _RNNTLossFunction.apply()
    │  reduction 处理（none / mean / sum）
    ▼
aclnn 层（C++ 两段式接口，pybind 绑定）
    │  aclnnRnntLossFusedGetWorkspaceSize()
    │  aclnnRnntLossFused()
    ▼
Host 侧（任务调度）
    │  最大任务优先 + 最小负载优先负载均衡
    │  将 B 条样本分配至多个 AI Core
    ▼
Kernel 侧（AscendC，单核处理单样本）
    │  Init：tile 切分、缓冲区初始化、缓存 targets/blank/clamp
    │  Process：
    │    1. log_softmax(logits) → log_prob（写入 grad 缓冲区）
    │    2. 对角线遍历计算 log_α
    │    3. 反对角线遍历计算 log_β
    │    4. 双路 loss 互校验
    │    5. 融合梯度公式计算 ∂L/∂h，原地覆写 grad
    └──────────────────────────────────────────
```

| 层级 | 职责 |
|---|---|
| Python | autograd 封装、reduction 聚合、blank 索引归一化 |
| aclnn | 两段式 C++ 接口，workspace 计算，pybind 导出 |
| Host | 核间负载均衡（最大任务优先 + 最小负载优先） |
| Kernel | 对角线遍历 α/β，tile 分块，log_prob / grad 内存复用 |

---

**调用说明**

```python
import torch
from rnnt_loss_ascend import rnnt_loss

# logits 为 2D 紧凑张量，按样本顺序拼接
# 样本 0: T0=4, U0=3, V=8  →  12 行
# 样本 1: T1=6, U1=2, V=8  →  12 行
# logits shape: (24, 8)
logits = torch.randn(24, 8, dtype=torch.float32, device="npu").requires_grad_(True)
targets = torch.tensor([[2, 3], [1, 0]], dtype=torch.int32, device="npu")
logit_lengths = torch.tensor([4, 6], dtype=torch.int32, device="npu")
target_lengths = torch.tensor([2, 1], dtype=torch.int32, device="npu")

loss = rnnt_loss(
    logits=logits,
    targets=targets,
    logit_lengths=logit_lengths,
    target_lengths=target_lengths,
    blank=-1,           # 使用词表最后一个 class 作为 blank
    clamp=-1.0,         # 不启用梯度截断
    reduction="mean",   # batch 维取均值
    fused_log_softmax=True  # kernel 内执行 log_softmax
)

loss.backward()
print("loss:", loss.item())
print("grad shape:", logits.grad.shape)  # (24, 8)
```

| 调用方式 | 样例入口 | 说明 |
|---|---|---|
| Python 接口 | `rnnt_loss_ascend.rnnt_loss()` | 与 torchaudio 对齐的上层接口，支持 autograd |
| aclnn 接口 | `test_aclnn_rnnt_loss_fused` | 通过 `aclnnRnntLossFused` 直接调用底层 kernel |

---

**算子特性**

| 特性 | 说明 |
|---|---|
| 前反向融合 | 一次 kernel 调用同时输出 loss 和 grad，避免二次遍历 α/β 表 |
| 内存原地复用 | log_prob 与 grad 共享同一 GM 缓冲区，峰值内存降低约 50% |
| 2D 紧凑输入 | 按样本拼接 logits，消除 padding 位置的无效 softmax 与梯度计算 |
| 对角线并行 | 反对角线上各 (t,u) 点依赖已满足，可在同一轮中批量载入 UB 处理 |
| 双路 loss 校验 | α 路径与 β 路径各自独立推导 loss 并互相比对，确保数值正确性 |
| 多核负载均衡 | 按 T×U 面积排序，优先分配大任务到空闲核，减少批次尾部等待 |
| 梯度截断保护 | clamp > 0 时对每个梯度分量执行 `grad = clamp(grad, -clamp, clamp)` |

---

**精度测试**

与 torchaudio 2.8.0 CPU 双精度参考实现对比（`reduction="mean"`）：

| 数据类型 | batch | T 范围 | U 范围 | V | loss 最大绝对误差 | grad 最大绝对误差 | 是否通过 |
|---|---|---|---|---|---|---|---|
| float32 | 2 | 100–1000 | 100–300 | 50 | < 1e-4 | < 1e-4 | ✓ |
| float32 | 4 | 200–2000 | 50–500 | 128 | < 1e-4 | < 1e-4 | ✓ |
| float16 | 2 | 100–500 | 50–200 | 64 | < 1e-3 | < 1e-3 | ✓ |
| float16 | 4 | 100–1000 | 50–300 | 128 | < 1e-3 | < 1e-3 | ✓ |

精度验证标准：float32 误差阈值 **1e-4**，float16 误差阈值 **1e-3**。

---

**性能数据**

与 torchaudio 2.8.0 CPU 双精度参考实现对比（`reduction="mean"`）：

| batch | T | U | V | torchaudio | NPU 融合算子 | 加速比 |
|---|---|---|---|---|---|---|
| 4 | 200 | 50 | 50 | 12.74 | 0.28 | 46.02 |
| 8 | 500 | 100 | 500 | 13.44 | 0.28 | 48.02 |
| 16 | 1000 | 200 | 5000 | 92.27 | 0.48 | 200.56 |
| 32 | 2000 | 500 | 50000 | 35993.41 | 8.22 | 4389.44 |