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
