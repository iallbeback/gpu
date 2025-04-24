#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef USE_DOUBLE
#define USE_DOUBLE 1
#endif

#if USE_DOUBLE
typedef double real_t;
#define MAXEPS 0.5
#define EPS_FORMAT "%14.7E"
#else
typedef float real_t;
#define MAXEPS 0.5f
#define EPS_FORMAT "%14.7E"
#endif

#ifndef L
#define L 900
#endif

#ifndef ITMAX
#define ITMAX 20
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif

__device__ static real_t MyatomicMax(real_t* address, real_t val) {
#if USE_DOUBLE
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
#endif
}

__global__ void compute_eps_and_update(real_t* __restrict A, const real_t* __restrict B, real_t* eps, int size) {
    __shared__ real_t shared_eps[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    
    real_t local_eps = 0.0;
    
    if (i < size - 1 && j < size - 1 && k < size - 1) {
        int idx = (k * size + j) * size + i;
#if USE_DOUBLE
        real_t tmp = fabs(B[idx] - A[idx]);
#else
        real_t tmp = fabsf(B[idx] - A[idx]);
#endif
        local_eps = tmp;
        A[idx] = B[idx];
    }
    
    shared_eps[tid] = local_eps;
    __syncthreads();

    for (int s = BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
#if USE_DOUBLE
            shared_eps[tid] = fmax(shared_eps[tid], shared_eps[tid + s]);
#else
            shared_eps[tid] = fmaxf(shared_eps[tid], shared_eps[tid + s]);
#endif
        }
        __syncthreads();
    }

    if (tid == 0) {
        MyatomicMax(eps, shared_eps[0]);
    }
}

__global__ void update_B(const real_t* __restrict A, real_t* __restrict B, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if (i < size - 1 && j < size - 1 && k < size - 1) {
        int idx = (k * size + j) * size + i;
        int stride = size * size;
        
        real_t sum = A[idx - stride] +  // (i-1,j,k)
                    A[idx - size] +     // (i,j-1,k)
                    A[idx - 1] +        // (i,j,k-1)
                    A[idx + 1] +        // (i,j,k+1)
                    A[idx + size] +     // (i,j+1,k)
                    A[idx + stride];    // (i+1,j,k)
        
        B[idx] = sum / 6.0f;
    }
}

void print_gpu_info() {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("GPU Device: %s\n", prop.name);
    printf("Total Global Memory: %.2f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Dim: %d x %d x %d\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
}

int main() {
    print_gpu_info();
    
    size_t size = L * L * L * sizeof(real_t);
    printf("Using array size: %d x %d x %d\n", L, L, L);
    printf("Memory used per array: %.2f MB\n", (L*L*L*sizeof(real_t)) / (1024.0*1024.0));
    
    real_t *h_A = (real_t*)malloc(size);
    real_t *h_B = (real_t*)malloc(size);
    
    if (!h_A || !h_B) {
        printf("Error: Memory allocation failed!\n");
        return 1;
    }
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                h_A[(i * L + j) * L + k] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1) {
                    h_B[(i * L + j) * L + k] = 0;
                } else {
                    h_B[(i * L + j) * L + k] = 4 + i + j + k;
                }
            }
        }
    }
    
    real_t *d_A, *d_B, *d_eps;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_eps, sizeof(real_t));
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (L + blockSize.x - 1) / blockSize.x,
        (L + blockSize.y - 1) / blockSize.y,
        (L + blockSize.z - 1) / blockSize.z
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int it = 1; it <= ITMAX; it++) {
        real_t h_eps = 0;
        cudaMemcpy(d_eps, &h_eps, sizeof(real_t), cudaMemcpyHostToDevice);
        
        compute_eps_and_update<<<gridSize, blockSize>>>(d_A, d_B, d_eps, L);
        update_B<<<gridSize, blockSize>>>(d_A, d_B, L);
        
        cudaMemcpy(&h_eps, d_eps, sizeof(real_t), cudaMemcpyDeviceToHost);
        
        printf(" IT = %4i   EPS = " EPS_FORMAT "\n", it, h_eps);
        if (h_eps < MAXEPS) break;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf(" Jacobi3D GPU Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", ITMAX);
    printf(" Time in seconds =       %12.2f\n", milliseconds / 1000.0f);
    printf(" Operation type  =     %s\n", (sizeof(real_t) == sizeof(double)) ? "double" : "float");
    printf(" GPU Memory used =     %.2f MB\n", (2.0 * L * L * L * sizeof(real_t)) / (1024.0 * 1024.0));
    printf(" Block size      =     %d x %d x %d\n", BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    printf(" END OF Jacobi3D Benchmark\n");
		
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_eps);
    free(h_A);
    free(h_B);
    
    return 0;
}