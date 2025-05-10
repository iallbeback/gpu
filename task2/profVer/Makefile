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

BLOCK_SIZES = 4x4 4x8 4x16 4x32 4x64 8x4 8x8 8x16 8x32 8x64 16x4 16x8 16x16 16x32 32x4 32x8 32x16 32x32

BUILD_TARGETS = adi3d_cpu adi3d_gpu compare_results

all: clean build run

build: $(BUILD_TARGETS)

adi3d_cpu: adi3d_cpu.c
	$(CC) $(CFLAGS) -o $@ $< -lm

adi3d_gpu: adi3d_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

compare_results: compare_results.c
	$(CC) $(COMPARE_FLAGS) -o $@ $< -lm

run: compare_results
	./compare_results

test_blocks: adi3d_gpu
	@echo "Testing different block sizes (all kernels use the same size)..."
	@mkdir -p test_blocks
	@for blocks in $(BLOCK_SIZES); do \
		x=$$(echo $$blocks | cut -d'x' -f1); \
		y=$$(echo $$blocks | cut -d'x' -f2); \
		echo "Testing block size: $$x $$y"; \
		./adi3d_gpu $$x $$y > test_blocks/gpu_$${blocks}.txt; \
	done

prof: adi3d_gpu
	@echo "Starting profiling runs, output will be saved to profiles/prof.txt..."
	@mkdir -p profiles
	@echo "--- Independent Block Profiling Results ---" > profiles/prof.txt
	@# Оставляем переменные для ясности в сообщениях, но используем литералы 16 16 в вызовах
	@FIXED_X=16
	@FIXED_Y=16

	@echo "" | tee -a profiles/prof.txt
	@echo "--- Varying x_sweep block size (others fixed at 16x16) ---" | tee -a profiles/prof.txt
	@echo "" | tee -a profiles/prof.txt
	@for blocks in $(BLOCK_SIZES); do \
		x_x=$$(echo $$blocks | cut -d'x' -f1); \
		x_y=$$(echo $$blocks | cut -d'x' -f2); \
		echo "Running with x_sweep block: $${blocks}" | tee -a profiles/prof.txt; \
		./adi3d_gpu $$x_x $$x_y 16 16 16 16 16 16 >> profiles/prof.txt 2>&1; \
		echo "--- End of run ---" >> profiles/prof.txt; \
		echo "" >> profiles/prof.txt; \
	done

	@echo "" | tee -a profiles/prof.txt
	@echo "--- Varying y_sweep block size (others fixed at 16x16) ---" | tee -a profiles/prof.txt
	@echo "" | tee -a profiles/prof.txt
	@for blocks in $(BLOCK_SIZES); do \
		y_x=$$(echo $$blocks | cut -d'x' -f1); \
		y_y=$$(echo $$blocks | cut -d'x' -f2); \
		echo "Running with y_sweep block: $${blocks}" | tee -a profiles/prof.txt; \
		./adi3d_gpu 16 16 $$y_x $$y_y 16 16 16 16 >> profiles/prof.txt 2>&1; \
		echo "--- End of run ---" >> profiles/prof.txt; \
		echo "" >> profiles/prof.txt; \
	done

	@echo "" | tee -a profiles/prof.txt
	@echo "--- Varying reorder block size (others fixed at 16x16) ---" | tee -a profiles/prof.txt
	@echo "" | tee -a profiles/prof.txt
	@for blocks in $(BLOCK_SIZES); do \
		r_x=$$(echo $$blocks | cut -d'x' -f1); \
		r_y=$$(echo $$blocks | cut -d'x' -f2); \
		echo "Running with reorder block: $${blocks}" | tee -a profiles/prof.txt; \
		./adi3d_gpu 16 16 16 16 $$r_x $$r_y 16 16 >> profiles/prof.txt 2>&1; \
		echo "--- End of run ---" >> profiles/prof.txt; \
		echo "" >> profiles/prof.txt; \
	done

	@echo "" | tee -a profiles/prof.txt
	@echo "--- Varying z_sweep block size (others fixed at 16x16) ---" | tee -a profiles/prof.txt
	@echo "" | tee -a profiles/prof.txt
	@for blocks in $(BLOCK_SIZES); do \
		z_x=$$(echo $$blocks | cut -d'x' -f1); \
		z_y=$$(echo $$blocks | cut -d'x' -f2); \
		echo "Running with z_sweep block: $${blocks}" | tee -a profiles/prof.txt; \
		./adi3d_gpu 16 16 16 16 16 16 $$z_x $$z_y >> profiles/prof.txt 2>&1; \
		echo "--- End of run ---" >> profiles/prof.txt; \
		echo "" >> profiles/prof.txt; \
	done

	@echo "Profiling runs completed. Results are in profiles/prof.txt"

clean:
	rm -f $(BUILD_TARGETS) *.o *.txt
	rm -rf test_blocks
	rm -rf profiles

.PHONY: all build clean run test_blocks prof
