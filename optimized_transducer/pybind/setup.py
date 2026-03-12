# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os, subprocess, sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExt(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(".")

class CMakeBuild(build_ext):
    def run(self):
        import torch
        for ext in self.extensions:
            outdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            build_dir = Path(self.build_temp) / ext.name
            build_dir.mkdir(parents=True, exist_ok=True)

            # 显式传递当前 Python 和 Torch 的路径信息
            cmake_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={outdir}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                f"-DTorch_DIR={torch.utils.cmake_prefix_path}/Torch",
            ]

            try:
                import torch_npu
                npu_path = os.path.dirname(torch_npu.__file__)
                cmake_args.append(f"-DNPU_PATH={npu_path}")
            except ImportError:
                pass

            subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_dir)
            subprocess.check_call(["cmake", "--build", ".", "-j4"], cwd=build_dir)

setup(
    name="optimized_transducer_ascend_ops",
    version="0.0.1",
    ext_modules=[CMakeExt("optimized_transducer_ascend_ops")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=["torch==2.8.0","torch_npu==2.8.0","torchaudio==2.8.0"],
)
