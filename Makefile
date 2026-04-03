# ============================================================
#  Makefile — SC4064 FlashAttention
#  Targets A100 (sm_80). Builds flash_attn and cublas_ref.
# ============================================================

NVCC     = nvcc
CFLAGS   = -O3 -arch=sm_80 --use_fast_math -Xcompiler -Wall
LDFLAGS  = -lcublas

.PHONY: all clean run ref

all: flash_attn cublas_ref

flash_attn: flash_attention.cu
	$(NVCC) $(CFLAGS) $< -o $@

cublas_ref: cublas_ref.cu
	$(NVCC) $(CFLAGS) $< $(LDFLAGS) -o $@

run: flash_attn cublas_ref
	./flash_attn
	./cublas_ref

ref: reference.py
	python3 reference.py

clean:
	rm -f flash_attn cublas_ref
