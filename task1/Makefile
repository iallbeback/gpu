NVCC = nvcc
CC = gcc

CFLAGS = -fopenmp -O3 -std=c99
NVCC_FLAGS = -Xcompiler "-fopenmp -O3" -std=c++11
COMPARE_FLAGS = -O3 -std=c99

TARGETS = jacobi3d_cpu jacobi3d_gpu compare_results

all: $(TARGETS)

jacobi3d_cpu: jacobi3d_cpu.c
	$(CC) $(CFLAGS) -o $@ $< -lm

jacobi3d_gpu: jacobi3d_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

compare_results: compare_results.c
	$(CC) $(COMPARE_FLAGS) -o $@ $< -lm

run: compare_results
	./compare_results

clean:
	rm -f $(TARGETS) *.o *.txt

.PHONY: all clean run
