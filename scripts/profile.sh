#!/bin/bash
# ============================================================
# profile.sh — Nsight Compute profiling for FlashAttention
# Run: bash profile.sh
# ============================================================

BINARY=./flash_attn

echo "=== Profiling Stage 0: Naive matmul_qk ==="
ncu --kernel-name naive_matmul_qk --launch-skip 3 --launch-count 1 \
    --metrics \
    gpu__time_duration.avg,\
    dram__bytes_read.sum,dram__bytes_write.sum,\
    dram__throughput.avg_pct_of_peak_sustained_elapsed,\
    sm__throughput.avg_pct_of_peak_sustained_elapsed \
    $BINARY 2>/dev/null | grep -E "(naive_matmul|gpu__|dram__|sm__)"

echo ""
echo "=== Profiling Stage 1: Fused v1 ==="
ncu --kernel-name flash_fused_v1 --launch-skip 3 --launch-count 1 \
    --metrics \
    gpu__time_duration.avg,\
    dram__bytes_read.sum,dram__bytes_write.sum,\
    dram__throughput.avg_pct_of_peak_sustained_elapsed,\
    sm__throughput.avg_pct_of_peak_sustained_elapsed,\
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
    smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct \
    $BINARY 2>/dev/null | grep -E "(flash_fused|gpu__|dram__|sm__|smsp__)"

echo ""
echo "=== Profiling Stage 2: Tiled v2 ==="
ncu --kernel-name flash_tiled_v2 --launch-skip 3 --launch-count 1 \
    --metrics \
    gpu__time_duration.avg,\
    dram__bytes_read.sum,dram__bytes_write.sum,\
    dram__throughput.avg_pct_of_peak_sustained_elapsed,\
    sm__throughput.avg_pct_of_peak_sustained_elapsed,\
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
    smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct \
    $BINARY 2>/dev/null | grep -E "(flash_tiled|gpu__|dram__|sm__|smsp__)"

echo ""
echo "=== Profiling Stage 3: wmma v3 (FULL) ==="
ncu --set full \
    --kernel-name flash_wmma_v3 --launch-skip 3 --launch-count 1 \
    $BINARY

echo ""
echo "=== Done ==="
