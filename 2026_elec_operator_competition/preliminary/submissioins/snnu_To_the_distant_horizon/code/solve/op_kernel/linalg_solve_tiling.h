/*
 * Copyright (c) 2026 zhiyuan (致远队).
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Tiling结构体定义的头文件 — solve
#pragma once

#include <cstdint>

struct LinalgSolveTilingData {
    uint32_t length;      // A 元素总数（模板保留字段）
    uint32_t n;           // 方阵维度
    uint32_t k;           // 右端列数（向量 B = 1）
    uint32_t batch;       // batch 项数
    uint32_t bStride;     // 每 batch 项的 B 元素步长（0 = 广播共享）
    uint32_t bVec;        // 1 = B 向量形式（输出 (batch,n)/(n,)）
    uint32_t pathS;       // 1 = UB 全量 LU 路径; 0 = GM 分块路径
    uint32_t irRounds;    // 迭代精化轮数
    uint32_t numCores;    // blockDim
    uint32_t isBf16;      // 1 = 输入 bf16（X 仍恒 fp32）
    uint32_t ubNeed;      // 内核 UB 池大小（floats）
    uint32_t sysFloat;    // workspace 用户区起始偏移（floats，= 系统保护区字节数/4）
    uint32_t extRes;      // 1 = 扩展精度残差（two-product + 补偿累加）；Path L 恒 1
};
