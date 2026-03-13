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

import torch
import custom_ops_lib


def unique_v3(input, return_inverse = False, return_counts = False):
    """
    对输入张量进行去重操作
    
    Args:
        input: 输入张量，支持 bfloat16, float16, int16, float, int32, int64 类型
        return_inverse: 是否返回逆映射，默认为 False
        return_counts: 是否返回每个唯一值的计数，默认为 False
    Returns:
        output: 去重后的唯一值（按升序排列）
        unique_cnt: 唯一值的数量
        inverse: 逆映射，默认为空
        counts: 每个唯一值的计数，默认为空
    """
    return custom_ops_lib.unique_v3(input, return_inverse, return_counts)
