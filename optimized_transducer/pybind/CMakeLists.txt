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

if(NOT Torch_DIR AND NOT ENV{Torch_DIR})
    message(STATUS "Torch_DIR not found, skipping pybind build during operator package compilation.")
    return()
endif()

cmake_minimum_required(VERSION 3.18)
project(optimized_transducer_ascend_ops)
set(CMAKE_CXX_STANDARD 17)

# Ascend 路径
set(ASCEND "$ENV{ASCEND_HOME_PATH}" CACHE PATH "")
if(NOT ASCEND)
    set(ASCEND "/usr/local/Ascend/latest")
endif()

# 使用 bisheng 编译器
if(EXISTS "${ASCEND}/compiler/ccec_compiler/bin/bisheng")
    set(CMAKE_CXX_COMPILER "${ASCEND}/compiler/ccec_compiler/bin/bisheng")
endif()

# Python & Torch (由 setup.py 传值)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)

# 模块后缀
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')" OUTPUT_VARIABLE PY_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(optimized_transducer_ascend_ops SHARED pybind_wrapper.cpp)

target_compile_definitions(optimized_transducer_ascend_ops PRIVATE TORCH_EXTENSION_NAME=optimized_transducer_ascend_ops)
target_compile_options(optimized_transducer_ascend_ops PRIVATE ${TORCH_CXX_FLAGS} -O2 -w)

target_include_directories(optimized_transducer_ascend_ops PRIVATE
    ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${NPU_PATH}/include
    ${ASCEND}/include ${ASCEND}/include/aclnn
    ${ASCEND}/opp/vendors/custom_nn/op_api/include/aclnnop)

target_link_directories(optimized_transducer_ascend_ops PRIVATE
    ${NPU_PATH}/lib ${ASCEND}/lib64 ${ASCEND}/opp/vendors/custom_nn/op_api/lib)

target_link_libraries(optimized_transducer_ascend_ops PRIVATE
    ${TORCH_LIBRARIES} torch_npu ascendcl nnopbase opapi cust_opapi)

set_target_properties(optimized_transducer_ascend_ops PROPERTIES
    PREFIX "" SUFFIX "${PY_SUFFIX}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ASCEND}/lib64;${NPU_PATH}/lib;${ASCEND}/opp/vendors/custom_nn/op_api/lib")
