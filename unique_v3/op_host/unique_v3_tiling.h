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

namespace optiling {
BEGIN_TILING_DATA_DEF(UniqueV3TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, shortBlockTileNum);
    TILING_DATA_FIELD_DEF(uint16_t, tileLength);
    TILING_DATA_FIELD_DEF(uint16_t, tailLength);
    TILING_DATA_FIELD_DEF(uint8_t, aivNum);
    TILING_DATA_FIELD_DEF(uint8_t, blockNum);
    TILING_DATA_FIELD_DEF(uint8_t, shortBlockNum);
    TILING_DATA_FIELD_DEF(bool, flagInverse);
    TILING_DATA_FIELD_DEF(bool, flagCounts);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UniqueV3, UniqueV3TilingData)
}