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

TARGETS = adi3d_cpu adi3d_gpu compare_results

BLOCK_SIZES = 4x4 8x8 16x16 32x32 16x8 8x16 32x16 16x32

all: $(TARGETS)

adi3d_cpu: adi3d_cpu.c
	$(CC) $(CFLAGS) -o $@ $< -lm

adi3d_gpu: adi3d_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

compare_results: compare_results.c
	$(CC) $(COMPARE_FLAGS) -o $@ $< -lm

run: compare_results
	./compare_results

test_blocks: adi3d_gpu
	@echo "Testing different block sizes..."
	@mkdir -p test_blocks
	@for blocks in $(BLOCK_SIZES); do \
		x=$$(echo $$blocks | cut -d'x' -f1); \
		y=$$(echo $$blocks | cut -d'x' -f2); \
		echo "Testing block size: $$x $$y"; \
		BLOCK_SIZE_X=$$x BLOCK_SIZE_Y=$$y ./adi3d_gpu > test_blocks/gpu_$${blocks}.txt; \
	done

clean:
	rm -f $(TARGETS) *.o *.txt

.PHONY: all clean run test_blocks