# ============================================================
#  Makefile — SC4064 FlashAttention
#  Targets A100 (sm_80). Builds flash_attn and cublas_ref.
# ============================================================
NVCC     = nvcc
CUTLASS_PATH = $(HOME)/cuda_kernals/cutlass
CFLAGS   = -O3 -arch=sm_80 --use_fast_math -std=c++17 -I$(CUTLASS_PATH)/include -Xcompiler -Wall
LDFLAGS  = -lcublas
.PHONY: all clean run ref
all: flash_attn cublas_ref
flash_attn: src/flash_attention.cu
	$(NVCC) $(CFLAGS) $< -o $@
cublas_ref: src/cublas_ref.cu
	$(NVCC) $(CFLAGS) $< $(LDFLAGS) -o $@
run: flash_attn cublas_ref
	./flash_attn
	./cublas_ref
ref: reference/reference.py
	python3 reference/reference.py
clean:
	rm -f flash_attn cublas_ref