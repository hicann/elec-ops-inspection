/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unique_v3_tiling_key.h
 * \brief unique_v3 tiling key declare
 */

#ifndef __UNIQUE_V3_TILING_KEY_H__
#define __UNIQUE_V3_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

/* Mode场景定义 */
#define UNIQUE_V3_SCH_MODE_FLOAT 0
#define UNIQUE_V3_SCH_MODE_INT32 1
#define UNIQUE_V3_SCH_MODE_FP16  2
#define UNIQUE_V3_SCH_MODE_BF16  3
#define UNIQUE_V3_SCH_MODE_INT16 4

/* 模板参数声明 */
ASCENDC_TPL_ARGS_DECL(
    UniqueV3,
    ASCENDC_TPL_UINT_DECL(schMode, 4, ASCENDC_TPL_UI_LIST,
                          UNIQUE_V3_SCH_MODE_FLOAT,
                          UNIQUE_V3_SCH_MODE_INT32,
                          UNIQUE_V3_SCH_MODE_FP16,
                          UNIQUE_V3_SCH_MODE_BF16,
                          UNIQUE_V3_SCH_MODE_INT16));

/* 模板参数组合 */
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                         UNIQUE_V3_SCH_MODE_FLOAT,
                         UNIQUE_V3_SCH_MODE_INT32,
                         UNIQUE_V3_SCH_MODE_FP16,
                         UNIQUE_V3_SCH_MODE_BF16,
                         UNIQUE_V3_SCH_MODE_INT16)));
#endif