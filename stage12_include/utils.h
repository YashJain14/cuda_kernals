#pragma once

namespace flash {

constexpr int constexpr_min(int a, int b) { return (a < b) ? a : b; }

constexpr int constexpr_max(int a, int b) { return (a > b) ? a : b; }

constexpr int constexpr_log2_floor(int n) { return 31 - __builtin_clz(n) - 1; }

constexpr int binary_to_pm1(int b) { return 2 * b - 1; }

} // namespace flash
