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

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} plugin_srcs)
add_library(cust_tf_parsers SHARED ${plugin_srcs})
target_compile_definitions(cust_tf_parsers PRIVATE google=ascend_private)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_tf_parsers PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_tf_parsers PRIVATE intf_pub graph)
install(TARGETS cust_tf_parsers
        LIBRARY DESTINATION packages/vendors/${vendor_name}/framework/tensorflow
)
