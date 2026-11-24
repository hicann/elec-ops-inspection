// Copyright (c) 2026 Qianqiuer.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
struct KeypointTiling { uint32_t batch, height, width, spatial; float tau; uint32_t stride, parts, reserved; };
