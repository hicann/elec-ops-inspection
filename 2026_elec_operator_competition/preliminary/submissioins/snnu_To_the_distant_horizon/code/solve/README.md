# solve

提交团队: 致远队
提交者: Purs


## 算子实现介绍

# linalg_solve · 线性修正方程求解算子

Ascend C 实现的 A·X = B 批量直解算子（fp32，昇腾 910B）。
面向电力系统状态估计 / 潮流计算的修正方程求解环节。

接口:  A [batch,n,n] + B [batch,n,k] → X [batch,n,k]
       n ≤ 512, k ≤ 256, batch ≤ 64, 题面保证 cond ≤ 1e3

算法:  LU 分解（部分选主元）+ 迭代精化（IR）
       三档路径按 UB 占用自动分派（范围判据，任一形状都有正确路径）
       S: n≤64 全驻留  |  M: 全驻留分块  |  L: GM 流式面板

精度:  逐元素 1e-4+1e-4|Xg| 且 ‖AX−B‖ ≤ 1e-4‖B‖，golden = fp64 numpy

实测:  910B2 单卡，12 个评测形状全 Pass（n=15~256, k=1~4, batch=1~4）

合规:  无 host 代算、无查表、无数据特化、无跨调用状态
