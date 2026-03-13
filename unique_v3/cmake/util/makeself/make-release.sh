#!/bin/sh

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

#
# Create a distributable archive of the current version of Makeself

VER=`cat VERSION`
mkdir -p /tmp/makeself-$VER release
cp -pPR makeself* test README.md COPYING VERSION .gitmodules /tmp/makeself-$VER/
./makeself.sh --notemp /tmp/makeself-$VER release/makeself-$VER.run "Makeself v$VER" echo "Makeself has extracted itself" 

