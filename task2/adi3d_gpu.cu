#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include <fstream>
#include <string>

struct KernelStats {
    float x_sweep_time = 0;
    float y_sweep_time = 0;
    float to_z_major_time = 0;
    float z_sweep_time = 0;
    float from_z_major_time = 0;
    float total_time = 0;
    int block_x = 0;
    int block_y = 0;
};

__device__ clock_t kernel_start_time;
__device__ clock_t kernel_end_time;

#ifdef USE_REAL_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

#define MAXEPS 0.01
#define EPS_FORMAT "%14.7E"

#ifndef NX
#define NX 900
#endif

#ifndef NY
#define NY 900
#endif

#ifndef NZ
#define NZ 900
#endif

#ifndef ITMAX
#define ITMAX 10
#endif

const int DEFAULT_BLOCK_X = 16;
const int DEFAULT_BLOCK_Y = 16;

#define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

template<typename T>
__device__ static T MyatomicMax(T* address, T val) {
    if (sizeof(T) == sizeof(float)) {
        int* address_as_int = (int*)address;
        int vl1 = *address_as_int, vl2;
        do {
            vl2 = vl1;
            vl1 = atomicCAS(address_as_int, vl2,
                __float_as_int(fmaxf(val, __int_as_float(vl2))));
        } while (vl2 != vl1);
        return __int_as_float(vl1);
    } else if (sizeof(T) == sizeof(double)) {
        unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
        unsigned long long int vl1 = *address_as_ull, vl2;
        do {
            vl2 = vl1;
            vl1 = atomicCAS(address_as_ull, vl2,
                __double_as_longlong(fmax(val, __longlong_as_double(vl2))));
        } while (vl2 != vl1);
        return __longlong_as_double(vl1);
    }
    return 0;
}

__global__ void x_sweep(real_t* a, int nx, int ny, int nz) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
        for (int i = 1; i < nx-1; i++) {
            int idx = i * ny * nz + j * nz + k;
            int idx_prev = (i-1) * ny * nz + j * nz + k;
            int idx_next = (i+1) * ny * nz + j * nz + k;

            a[idx] = (a[idx_prev] + a[idx_next]) / 2;
        }
    }
}

__global__ void y_sweep(real_t* a, int nx, int ny, int nz) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= 1 && i < nx-1 && k >= 1 && k < nz-1) {
        for (int j = 1; j < ny-1; j++) {
            int idx = i * ny * nz + j * nz + k;
            int idx_prev = i * ny * nz + (j-1) * nz + k;
            int idx_next = i * ny * nz + (j+1) * nz + k;

            a[idx] = (a[idx_prev] + a[idx_next]) / 2;
        }
    }
}

__global__ void reorder_to_z_major(real_t* a, real_t* b, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        for (int k = 0; k < nz; k++) {
            int idx_a = k * ny * nz + i * nz + j;         // [i][j][k]
            int idx_b = j * nx * ny + k * ny + i;         // [k][i][j]
            b[idx_b] = a[idx_a];
        }
    }
}

__global__ void z_sweep(real_t* b, int nx, int ny, int nz, real_t* d_eps) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    real_t local_eps = 0;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        for (int k = 1; k < nz - 1; k++) {
            int idx      = k * nx * ny + i * ny + j;     // b[k][i][j]
            int idx_prev = (k - 1) * nx * ny + i * ny + j;
            int idx_next = (k + 1) * nx * ny + i * ny + j;

            real_t tmp1 = (b[idx_prev] + b[idx_next]) / 2;
            real_t tmp2 = fabs(b[idx] - tmp1);
            local_eps = max(local_eps, tmp2);
            b[idx] = tmp1;
        }
    }

    extern __shared__ real_t shared_eps[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    shared_eps[tid] = local_eps;
    __syncthreads();

    for (int s = blockDim.x * blockDim.y  / 2; s > 0; s /= 2) {
        if (tid < s) {
            shared_eps[tid] = max(shared_eps[tid], shared_eps[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        MyatomicMax(d_eps, shared_eps[0]);
    }
}

__global__ void reorder_from_z_major(real_t* b, real_t* a, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        for (int k = 0; k < nz; k++) {
            int idx_b = i * nx * ny + k * ny + j;         // [k][i][j]
            int idx_a = k * ny * nz + j * nz + i;         // [i][j][k]
            a[idx_a] = b[idx_b];
        }
    }
}

void init(real_t *a, int nx, int ny, int nz) {
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) {
                int idx = i * ny * nz + j * nz + k;
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a[idx] = (real_t)10.0 * i / (nx - 1) + (real_t)10.0 * j / (ny - 1) + (real_t)10.0 * k / (nz - 1);
                else
                    a[idx] = (real_t)0;
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

void write_kernel_stats(const KernelStats& stats, const std::string& filename) {
    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        printf("Failed to open profile file\n");
        return;
    }
    
    out << "Block size: " << stats.block_x << "x" << stats.block_y << "\n";
    out << "x_sweep: " << stats.x_sweep_time << " ms\n";
    out << "y_sweep: " << stats.y_sweep_time << " ms\n";
    out << "to_z_major: " << stats.to_z_major_time << " ms\n";
    out << "z_sweep: " << stats.z_sweep_time << " ms\n";
    out << "from_z_major: " << stats.from_z_major_time << " ms\n";
    out << "Total: " << stats.total_time << " ms\n\n";
    out.close();
}

int main(int argc, char** argv) {
    int block_x = DEFAULT_BLOCK_X;
    int block_y = DEFAULT_BLOCK_Y;
    
    if (argc == 3) {
        block_x = atoi(argv[1]);
        block_y = atoi(argv[2]);
    }
    
    dim3 blockSize(block_x, block_y);
    
    print_gpu_info();
    
    int nx = NX, ny = NY, nz = NZ;
    printf("Using array size: %d x %d x %d\n", nx, ny, nz);
    printf("Memory used per array: %.2f MB\n", (nx*ny*nz*sizeof(real_t)) / (1024.0*1024.0));
    
    real_t *h_A = (real_t*)malloc(nx * ny * nz * sizeof(real_t));
    if (!h_A) {
        printf("Host memory allocation failed\n");
        return 1;
    }
    
    init(h_A, nx, ny, nz);
    
    real_t *d_A, *d_B, *d_eps;
    CUDA_CHECK(cudaMalloc(&d_A, nx * ny * nz * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_B, nx * ny * nz * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_eps, sizeof(real_t)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nx * ny * nz * sizeof(real_t), cudaMemcpyHostToDevice));
    
    dim3 gridSizeX((ny + blockSize.x - 1) / blockSize.x, (nz + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeY((nx + blockSize.x - 1) / blockSize.x, (nz + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeZ((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    KernelStats stats;
    stats.block_x = block_x;
    stats.block_y = block_y;
    
    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    
    for (int it = 1; it <= ITMAX; it++) {
        real_t h_eps = 0;
        CUDA_CHECK(cudaMemcpy(d_eps, &h_eps, sizeof(real_t), cudaMemcpyHostToDevice));
        
        // x_sweep
        CUDA_CHECK(cudaEventRecord(start_kernel));
        x_sweep<<<gridSizeX, blockSize>>>(d_A, nx, ny, nz);
        CUDA_CHECK(cudaEventRecord(stop_kernel));
        CUDA_CHECK(cudaEventSynchronize(stop_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&stats.x_sweep_time, start_kernel, stop_kernel));
        
        // y_sweep
        CUDA_CHECK(cudaEventRecord(start_kernel));
        y_sweep<<<gridSizeY, blockSize>>>(d_A, nx, ny, nz);
        CUDA_CHECK(cudaEventRecord(stop_kernel));
        CUDA_CHECK(cudaEventSynchronize(stop_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&stats.y_sweep_time, start_kernel, stop_kernel));
        
        // reorder_to_z_major
        CUDA_CHECK(cudaEventRecord(start_kernel));
        reorder_to_z_major<<<gridSizeZ, blockSize>>>(d_A, d_B, nx, ny, nz);
        CUDA_CHECK(cudaEventRecord(stop_kernel));
        CUDA_CHECK(cudaEventSynchronize(stop_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&stats.to_z_major_time, start_kernel, stop_kernel));
        
        // z_sweep
        CUDA_CHECK(cudaEventRecord(start_kernel));
        size_t shared_size = blockSize.x * blockSize.y * sizeof(real_t);
		z_sweep<<<gridSizeZ, blockSize, shared_size>>>(d_B, nx, ny, nz, d_eps);
        CUDA_CHECK(cudaEventRecord(stop_kernel));
        CUDA_CHECK(cudaEventSynchronize(stop_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&stats.z_sweep_time, start_kernel, stop_kernel));
        
        // reorder_from_z_major
        CUDA_CHECK(cudaEventRecord(start_kernel));
        reorder_from_z_major<<<gridSizeZ, blockSize>>>(d_B, d_A, nx, ny, nz);
        CUDA_CHECK(cudaEventRecord(stop_kernel));
        CUDA_CHECK(cudaEventSynchronize(stop_kernel));
        CUDA_CHECK(cudaEventElapsedTime(&stats.from_z_major_time, start_kernel, stop_kernel));
        
        CUDA_CHECK(cudaMemcpy(&h_eps, d_eps, sizeof(real_t), cudaMemcpyDeviceToHost));
        
        printf(" IT = %4i   EPS = " EPS_FORMAT "\n", it, h_eps);
        if (h_eps < MAXEPS) break;
    }
	
	stats.total_time = stats.x_sweep_time + 
					   stats.y_sweep_time + 
					   stats.to_z_major_time + 
					   stats.z_sweep_time + 
					   stats.from_z_major_time;
    
    std::string profile_dir = "profiles";
    system(("mkdir -p " + profile_dir).c_str());
    std::string profile_file = profile_dir + "/gpu_profile.txt";
    write_kernel_stats(stats, profile_file);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", ITMAX);
    printf(" Time in seconds =       %12.2f\n", milliseconds / 1000.0f);
    printf(" Operation type  =     %s\n", (sizeof(real_t) == sizeof(double)) ? "double" : "float");
    printf(" GPU Memory used =     %.2f MB\n", (2.0 * nx * ny * nz * sizeof(real_t)) / (1024.0 * 1024.0));
    printf(" Block size      =     %d x %d\n", blockSize.x, blockSize.y);
    printf(" END OF ADI Benchmark\n");
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_eps));
    free(h_A);

    return 0;
}