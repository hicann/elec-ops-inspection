#!/bin/bash

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

project_path=$1
build_path=$2
vendor_name=customize
if [[ ! -d "$project_path" ]]; then
    echo "[ERROR] No projcet path is provided"
    exit 1
fi

if [[ ! -d "$build_path" ]]; then
    echo "[ERROR] No build path is provided"
    exit 1
fi

# copy aicpu kernel so operators
if [[ -d "${project_path}/cpukernel/aicpu_kernel_lib" ]]; then
    cp -f ${project_path}/cpukernel/aicpu_kernel_lib/* ${build_path}/makepkg/packages/vendors/$vendor_name/op_impl/cpu/aicpu_kernel/impl
    rm -rf ${project_path}/cpukernel/aicpu_kernel_lib
fi
