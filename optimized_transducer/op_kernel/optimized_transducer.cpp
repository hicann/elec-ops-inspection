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

template <uint32_t schMode>
__global__ __aicore__ void optimized_transducer(GM_ADDR logits, GM_ADDR targets, GM_ADDR logitLengths,
                                                 GM_ADDR targetLengths, GM_ADDR loss, GM_ADDR grad,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(OptimizedTransducerTilingData);
    GET_TILING_DATA_WITH_STRUCT(OptimizedTransducerTilingData, tilingData, tiling);

    NsOptimizedTransducer::OptimizedTransducer op;
    op.Init(logits, targets, logitLengths, targetLengths, loss, grad, workspace, &tilingData);
    op.Process();
}