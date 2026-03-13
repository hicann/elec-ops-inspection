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

extern "C" __global__ __aicore__ void unique_v3(
    GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt, 
    GM_ADDR inverse, GM_ADDR counts, 
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelUnique<DTYPE_INPUT> op(pipe);
    op.Init(input,
            output,
            uniqueCnt,
            inverse,
            counts,
            workspace,
            tiling_data.totalLength,
            tiling_data.shortBlockTileNum,
            tiling_data.tileLength,
            tiling_data.tailLength,
            tiling_data.aivNum,
            tiling_data.blockNum,
            tiling_data.shortBlockNum,
            tiling_data.flagInverse,
            tiling_data.flagCounts);
    op.Process();
}