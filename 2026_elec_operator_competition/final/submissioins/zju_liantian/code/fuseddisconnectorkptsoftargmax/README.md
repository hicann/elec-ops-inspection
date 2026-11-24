# fuseddisconnectorkptsoftargmax

提交团队: 炼天魔尊
提交者: 钱瑞言

## 算子实现介绍

基于 Ascend C 在 Ascend 910B 上实现隔离开关关键点坐标和置信度分数融合解码算子 FusedDisconnectorKptSoftArgmax（v411）。Host 侧按输入几何 (B, H, W) 与缩放系数 tau/stride 静态分派多条 Cube/Vector 融合路径及通用 FP32 兜底路径：

- **通用 FP32 向量兜底路径（ROUTE 0）**：每核独立处理单个 `(b, k)` 完整热图，支持任意合法输入与自动回退，避免完整概率图写回 Global Memory。
- **零权重单核快检路径（ROUTE 1）**：先检测全部 1152 个卷积权重是否为零；全零时直接写出半像素中心坐标与 bias 分数，非零时回退通用 FP32 路径。
- **片上补偿双 FP16 重组路径（ROUTE 2 / ROUTE 3）**：覆盖 `64x64` 大批量及 `72x96` 多批量场景，Cube 完成垂直投影与片上补偿双 FP16 水平重组，AIV 归约局部统计并融合输出；若检测到异常数值则自动回退通用 FP32 路径。
- **直接垂直投影与行归约路径（ROUTE 4 / ROUTE 5）**：覆盖 `B=2` 中小尺度，Cube 双缓冲 L0C 计算 12 路垂直投影，AIV 单次 Gather 完成水平零填充移位累加与行级 `WholeReduceSum` 归约。
- **AIV 通道块缓存路径（ROUTE 6）**：覆盖 `B=3, 48x64`，将 12 个平面拆分为 36 个行条带在片上完成 3x3 卷积与局部统计。
- **小批量 Cube 垂直投影路径（ROUTE 7 / ROUTE 8）**：覆盖 `B=1, 48x64` 与 `B=1, 64x64`，按行分片融合 Cube 垂直投影、AIV 权重重排及向量 Gather 归约（Cube 调度段经授权局部复用并注明来源）。
- **三链 C8 空间流水路径（ROUTE 9）**：覆盖 `B=1, 72x96`，以 8 通道打包与三条独立 `dx` 累加链完成卷积与分块统计融合。

最终正式提交 Pass（37.74 分，第 7 名），15 组测试点及 64 组覆盖用例全部通过。

## 算子编译

安装 CANN Toolkit 9.0 或兼容版本及 CMake，然后执行：

```bash
source /path/to/cann/set_env.sh
cmake -S . -B build -DCANNJUDGE_COMPILE_JOBS=4
cmake --build build --parallel 4 --target binary package
```
