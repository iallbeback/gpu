NVCC = nvcc
CC = gcc

USE_REAL_DOUBLE ?= 1
ifeq ($(USE_REAL_DOUBLE), 1)
PRECISION_FLAG = -DUSE_REAL_DOUBLE
else
PRECISION_FLAG =
endif

CFLAGS = -fopenmp -O3 -std=c99 $(PRECISION_FLAG)
NVCC_FLAGS = -Xcompiler "-fopenmp -O3" -std=c++11 $(PRECISION_FLAG)
COMPARE_FLAGS = -O3 -std=c99

TARGETS = jacobi3d_cpu jacobi3d_gpu compare_results

BLOCK_SIZES = 4x4x4 8x8x8 16x16x16 32x32x1 16x8x4 8x16x4 4x16x8 32x4x4 16x4x4 8x8x4 4x4x8 2x2x2 4x8x8 8x4x8 8x8x4 16x16x1 32x8x1 64x4x1

all: $(TARGETS)

jacobi3d_cpu: jacobi3d_cpu.c
	$(CC) $(CFLAGS) -o $@ $< -lm

jacobi3d_gpu: jacobi3d_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

compare_results: compare_results.c
	$(CC) $(COMPARE_FLAGS) -o $@ $< -lm

run: compare_results
	./compare_results

test_blocks: jacobi3d_gpu
	@echo "Testing different block sizes..."
	@for blocks in $(BLOCK_SIZES); do \
		x=$$(echo $$blocks | cut -d'x' -f1); \
		y=$$(echo $$blocks | cut -d'x' -f2); \
		z=$$(echo $$blocks | cut -d'x' -f3); \
		echo "Testing block size: $$x $$y $$z"; \
		./jacobi3d_gpu $$x $$y $$z | tee test_blocks/gpu_$${blocks}.txt; \
	done

clean:
	rm -f $(TARGETS) *.o *.txt

.PHONY: all clean run test_blocks