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

struct OptimizedTransducerTilingData {
    int64_t tileNum;
    int64_t blank;
    float clamp;
    bool fusedLogSoftmax;     
    int64_t totalPositions;   // 总位置数（点位数量），Σ(t,u)
    int64_t vocabSize;        // K，词表大小
    int64_t batchSize;         // N，批次大小
    int64_t maxTargetLength;   // targets 的最大长度（从 shape 获取）
    uint64_t ubSize;           // UB内存大小，单位：字节
    uint32_t blockSize;
    uint32_t usedCoreNum;      // 显式记录计划使用的核心数
};

#endif