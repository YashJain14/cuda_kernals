#pragma once

namespace flash {

#include "common.h"

FA_HOST_DEVICE_CONSTEXPR int constexpr_min(int a, int b) { return (a < b) ? a : b; }

FA_HOST_DEVICE_CONSTEXPR int constexpr_max(int a, int b) { return (a > b) ? a : b; }

FA_HOST_DEVICE_CONSTEXPR int constexpr_log2_floor(int n) { return 31 - __builtin_clz(n) - 1; }

FA_HOST_DEVICE_CONSTEXPR int binary_to_pm1(int b) { return 2 * b - 1; }

} // namespace flash
