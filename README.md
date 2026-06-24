<p align="center">
  <img src="https://img.shields.io/badge/Electrical%20Engineering%20SIG-CANN%20Community-orange" />
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" />
  <img src="https://img.shields.io/badge/status-active-brightgreen" />
</p>

> **elec-ops-inspection** 是 CANN 社区 Electrical Engineering SIG（电力行业兴趣小组）旗下的电力装备巡检算子库，
> 覆盖 CV 视觉检测与具身智能两大技术路线，面向输电线路、变电设备、配电设施等电力装备的智能化巡检场景，
> 基于华为昇腾（Ascend）硬件平台进行深度优化。

---

## 项目背景

电力装备巡检是保障电网安全稳定运行的核心运维环节。传统人工巡检面临效率低、成本高、高危场景作业风险大等问题。本仓库聚焦两大技术方向：

**① CV 视觉检测**：面向输电通道隐患识别，包括大型设备作业、鸟巢、绝缘子破损、锈蚀等典型缺陷的高精度检测，基于 Co-DETR 等先进目标检测模型，在 910B 上实现推理性能从**秒（s）级到毫秒（ms）级**的跨越。

**② 具身智能巡检**：面向机器狗、无人机等具身智能形态的"地空协同"移动巡检场景，提供面向大规模空间尺度建模（3DGS）的昇腾专用算子，支撑换流站等大型设施的空间感知与智能巡检。

---

## 核心算子列表

| 算子名称                    | 场景     | 描述               | 状态     |
| ----------------------- | ------ | ---------------- | ------ |
| `optimized_transducer` | 语音识别 | 对`RNN-T`模型的`loss`算子和`softmax`算子做了融合，显存优化和性能优化 | ✅ 已发布  |
| `unique_v3` | 数学通用算子  | 对排序算法在AIV上做了优化，增加`index` `counter`输出   | ✅ 已发布 |

---

## 开发辅助

| Skill | 用途 |
|---|---|
| [`electric-inspection-shape-boundary`](.agents/skills/electric-inspection-shape-boundary/SKILL.md) | 从接口、Host Tiling、Kernel 和样例推导 Shape、rank、tile/block、尾块、对齐、packed 输入关系与整数范围边界，生成有公式依据的最小边界用例。 |

---

## 🤝 参与贡献

欢迎所有对电力 AI 和昇腾开发感兴趣的开发者参与共建！贡献方式：

1. 提交 Issue 反馈问题或建议
2. Fork 本仓库并提交 Pull Request
3. 参与 Electrical Engineering SIG 定期研讨会
4. 完善文档与示例

贡献规范请参考 [`CONTRIBUTING.md`]()。
新增或重写算子说明时，可参考 [`docs/operator-readme-template.md`](docs/operator-readme-template.md)。

仓库提供了面向算子性能证据整理的辅助 Skill：[`inspection-benchmark-evidence`](.agents/skills/inspection-benchmark-evidence/SKILL.md)，用于在提交 PR 前检查巡检算子的 benchmark 环境、shape 矩阵、baseline、统计口径和可复现命令。

仓库提供了面向技能贡献的辅助 Skill：[`electric-inspection-skill-creator`](.agents/skills/electric-inspection-skill-creator/SKILL.md)，用于为电力巡检算子场景创建差异化、可验证、可提 PR 的仓库本地 Skill。

---

## 👥 维护团队

**Maintainers**

|姓名|GitCode ID|单位|
|---|---|---|
|梁寿愚|@jason2025|南方电网人工智能研究中心|
|陆璐|@Lulu_scut|华南理工大学|
|陈辰|@xchencehn|杭州天宽科技|
|张玉橙|@Splendid2025|昇腾产品线|
|田野|@tianye525|AI算力基础设施|

**Committers**

|姓名|GitCode ID|单位|
|---|---|---|
|余涛|@yutao_scut|华南理工大学电力学院|
|刘迪|@weixin_34344963|清华大学电机系|
|江豪|@yhyyyl|华南理工大学 / 南方电网联培|
|陈昀|@edconeone|华南理工大学|
|莫程翔|@Andrewmo1|华为公司|
|李博|@gcw_FHrfwZBn|华为公司|

---

## 📄 许可证

本项目基于 [Apache License 2.0]() 开源。
