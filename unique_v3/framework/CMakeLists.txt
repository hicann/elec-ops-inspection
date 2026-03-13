# Copyright 2026 Electrical Engineering SIG - CANN Community
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
