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

message(STATUS "TILING SINK TASK BEGIN")
message(STATUS "TARGET: ${TARGET}")
message(STATUS "OPTION: ${OPTION}")
message(STATUS "SRC: ${SRC}")
message(STATUS "VENDOR: ${VENDOR_NAME}")

set(CMAKE_CXX_COMPILER ${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
set(CMAKE_C_COMPILER ${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc)

string(REPLACE " " ";" SRC "${SRC}")
add_library(${TARGET} ${OPTION}
    ${SRC}
)
target_compile_definitions(${TARGET} PRIVATE
    DEVICE_OP_TILING_LIB
    _FORTIFY_SOURCE=2
    google=ascend_private
)
target_include_directories(${TARGET} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
)
target_compile_options(${TARGET} PRIVATE
    -fPIC
    -fstack-protector-strong
    -fstack-protector-all
    -O2
    -std=c++11
    -fvisibility-inlines-hidden
    -fvisibility=hidden
)
target_link_libraries(${TARGET} PRIVATE
    -Wl,--whole-archive
    device_register
    c_sec
    mmpa
    tiling_api
    platform_static
    ascend_protobuf
    exe_meta_device
    aicpu_cust_log
    -Wl,--no-whole-archive
)
target_link_directories(${TARGET} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/device/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/lib64
)
set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME cust_opmaster
)