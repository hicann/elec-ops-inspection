#!/usr/bin/env python

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

# -*- coding: UTF-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import sys
import os
import opdesc_parser

PYF_PATH = os.path.dirname(os.path.realpath(__file__))


class OpInfo:
    def __init__(self: any, op_type: str, cfg_file: str):
        op_descs = opdesc_parser.get_op_desc(
            cfg_file, [], [], opdesc_parser.OpDesc, [op_type]
        )
        if op_descs is None or len(op_descs) != 1:
            raise RuntimeError("cannot get op info of {}".format(op_type))
        self.op_desc = op_descs[0]

    def get_op_file(self: any):
        return self.op_desc.op_file

    def get_op_intf(self: any):
        return self.op_desc.op_intf

    def get_inputs_name(self: any):
        return self.op_desc.input_ori_name

    def get_outputs_name(self: any):
        return self.op_desc.output_ori_name


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        raise RuntimeError("arguments must greater than 2")
    op_info = OpInfo(sys.argv[1], sys.argv[2])
    print(op_info.get_op_file())
    print(op_info.get_op_intf())
    print(op_info.get_inputs_name())
    print(op_info.get_outputs_name())
