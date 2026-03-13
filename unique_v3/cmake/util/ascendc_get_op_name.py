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

import configparser
import argparse


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--ini-file", help="op info ini."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parse()
    op_config = configparser.ConfigParser()
    op_config.read(args.ini_file)
    for section in op_config.sections():
        print(section, end="-")
        print(op_config.get(section, "opFile.value"), end="\n")