# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mindspore")
    if(EXISTS  "${CMAKE_CURRENT_SOURCE_DIR}/caffe_plugin")
        add_subdirectory(caffe_plugin)
    endif()
    if(EXISTS  "${CMAKE_CURRENT_SOURCE_DIR}/tf_plugin")
        add_subdirectory(tf_plugin)
    endif()
    if(EXISTS  "${CMAKE_CURRENT_SOURCE_DIR}/onnx_plugin")
        add_subdirectory(onnx_plugin)
    endif()
endif()
