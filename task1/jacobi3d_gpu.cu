#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>

#ifdef USE_REAL_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

#define MAXEPS 0.5
#define EPS_FORMAT "%14.7E"

#ifndef L
#define L 900
#endif

#ifndef ITMAX
#define ITMAX 20
#endif

#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 16
#endif

#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 4
#endif

#ifndef BLOCK_SIZE_Z
#define BLOCK_SIZE_Z 4
#endif

#define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

template <typename T>
__device__ static T MyatomicMax(T* address, T val);

template <typename T>
__device__ T MyatomicMax(T* address, T val) {
    if (sizeof(T) == sizeof(double)) {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
        } while (assumed != old);
        return __longlong_as_double(old);
    } else {
        int* address_as_int = (int*)address;
        int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }
}

__global__ void compute_eps_and_update(const real_t* __restrict A, const real_t* __restrict B, real_t* eps, int size) {
    __shared__ real_t shared_max[BLOCK_SIZE_X*BLOCK_SIZE_Y*BLOCK_SIZE_Z / 32];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    real_t my_max = 0.0;
    
    if (i < size - 1 && j < size - 1 && k < size - 1) {
        int idx = (k * size + j) * size + i;
        my_max = fabs(B[idx] - A[idx]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        real_t tmp = __shfl_down_sync(0xFFFFFFFF, my_max, offset);
        my_max = fmax(my_max, tmp);
    }

    if (lane_id == 0) {
        shared_max[warp_id] = my_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        my_max = (lane_id < blockDim.x * blockDim.y * blockDim.z / 32) ? shared_max[lane_id] : 0.0;
        for (int offset = 4; offset > 0; offset >>= 1) {
            real_t tmp = __shfl_down_sync(0xFFFFFFFF, my_max, offset);
            my_max = fmax(my_max, tmp);
        }
        if (lane_id == 0) {
            MyatomicMax(eps, my_max);
        }
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
        
        B[idx] = sum / (real_t)6.0;
    }
}

void print_gpu_info() {
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU Device: %s\n", prop.name);
    printf("Total Global Memory: %.2f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Dim: %d x %d x %d\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
}

int main(int argc, char** argv) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    
    if (argc > 3) {
        blockSize.x = atoi(argv[1]);
        blockSize.y = atoi(argv[2]);
        blockSize.z = atoi(argv[3]);
        printf("Using custom block size: %d x %d x %d\n", blockSize.x, blockSize.y, blockSize.z);
    }
    
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
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_eps, sizeof(real_t)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    dim3 gridSize(
        (L + blockSize.x - 1) / blockSize.x,
        (L + blockSize.y - 1) / blockSize.y,
        (L + blockSize.z - 1) / blockSize.z
    );

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int it = 1; it <= ITMAX; it++) {
        real_t h_eps = 0;
        CUDA_CHECK(cudaMemcpy(d_eps, &h_eps, sizeof(real_t), cudaMemcpyHostToDevice));
        
        if (it % 2 == 1) {
            compute_eps_and_update<<<gridSize, blockSize>>>(d_A, d_B, d_eps, L);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            update_B<<<gridSize, blockSize>>>(d_B, d_A, L);
        } else {
            compute_eps_and_update<<<gridSize, blockSize>>>(d_B, d_A, d_eps, L);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            update_B<<<gridSize, blockSize>>>(d_A, d_B, L);
        }
        
        CUDA_CHECK(cudaMemcpy(&h_eps, d_eps, sizeof(real_t), cudaMemcpyDeviceToHost));
        
        printf(" IT = %4i   EPS = " EPS_FORMAT "\n", it, h_eps);
        if (h_eps < MAXEPS) break;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf(" Jacobi3D GPU Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", ITMAX);
    printf(" Time in seconds =       %12.2f\n", milliseconds / 1000.0f);
    printf(" Operation type  =     %s\n", (sizeof(real_t) == sizeof(double)) ? "double" : "float");
    printf(" GPU Memory used =     %.2f MB\n", (2.0 * L * L * L * sizeof(real_t)) / (1024.0 * 1024.0));
    printf(" Block size      =     %d x %d x %d\n", blockSize.x, blockSize.y, blockSize.z);
    printf(" END OF Jacobi3D Benchmark\n");
        
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_eps));
    free(h_A);
    free(h_B);
    
    return 0;
}