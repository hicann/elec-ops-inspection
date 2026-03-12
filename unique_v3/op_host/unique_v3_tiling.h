/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/tilingdata_base.h"

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