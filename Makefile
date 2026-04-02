# FlashAttention Project — Makefile for A100 (sm_80)

NVCC      = nvcc
NVFLAGS   = -O3 -arch=sm_80 --use_fast_math -Xcompiler -Wall
TARGET    = flash_attn

# Nsight Compute profiling target
PROFILE_BINARY = ./$(TARGET)

all: $(TARGET)

$(TARGET): flash_attention.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# Run benchmark
run: $(TARGET)
	./$(TARGET)

# Run PyTorch reference (requires torch installed)
ref:
	python3 reference.py

# Profile with Nsight Compute (Stage 3 kernel only)
profile: $(TARGET)
	ncu --set full \
	    --kernel-name flash_wmma_v3 \
	    --launch-skip 3 --launch-count 1 \
	    $(PROFILE_BINARY)

# Profile all kernels (brief)
profile-all: $(TARGET)
	ncu --metrics \
	    gpu__time_duration.avg,\
	    sm__throughput.avg_pct_of_peak_sustained_elapsed,\
	    dram__throughput.avg_pct_of_peak_sustained_elapsed,\
	    smsp__warps_active.avg_pct_of_peak_sustained_elapsed \
	    $(PROFILE_BINARY)

clean:
	rm -f $(TARGET)

.PHONY: all run ref profile profile-all clean
