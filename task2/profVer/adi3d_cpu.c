#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

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

#ifndef MAXEPS
#define MAXEPS 0.01
#endif

#define EPS_FORMAT "%14.7E"

void print_gpu_info() {
    printf("CPU Threads available = %d\n", omp_get_max_threads());
}

int main() {
    int nx = NX, ny = NY, nz = NZ;
    double *a;
    double startt, endt;
    
    print_gpu_info();
    printf("Using array size: %d x %d x %d\n", nx, ny, nz);
    printf("Memory used: %.2f MB\n", (nx*ny*nz*sizeof(double)) / (1024.0*1024.0));
    
    a = (double*)malloc(nx * ny * nz * sizeof(double));
    if (!a) {
        printf("Error: Memory allocation failed!\n");
        return 1;
    }

    // Initialize array
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    a[i*ny*nz + j*nz + k] = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                else
                    a[i*ny*nz + j*nz + k] = 0;
            }
        }
    }

    startt = omp_get_wtime();

    for (int it = 1; it <= ITMAX; it++) {
        double eps = 0;
        
        #pragma omp parallel for collapse(2)
        for (int j = 1; j < ny - 1; j++) {
            for (int k = 1; k < nz - 1; k++) {
                for (int i = 1; i < nx - 1; i++) {
                    a[i*ny*nz + j*nz + k] = (a[(i-1)*ny*nz + j*nz + k] + a[(i+1)*ny*nz + j*nz + k]) / 2;
                }
            }
        }

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < nx - 1; i++) {
            for (int k = 1; k < nz - 1; k++) {
                for (int j = 1; j < ny - 1; j++) {
                    a[i*ny*nz + j*nz + k] = (a[i*ny*nz + (j-1)*nz + k] + a[i*ny*nz + (j+1)*nz + k]) / 2;
                }
            }
        }

        #pragma omp parallel for collapse(2) reduction(max:eps)
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                for (int k = 1; k < nz - 1; k++) {
                    double tmp1 = (a[i*ny*nz + j*nz + k-1] + a[i*ny*nz + j*nz + k+1]) / 2;
                    double tmp2 = fabs(a[i*ny*nz + j*nz + k] - tmp1);
                    eps = Max(eps, tmp2);
                    a[i*ny*nz + j*nz + k] = tmp1;
                }
            }
        }

        printf(" IT = %4i   EPS = " EPS_FORMAT "\n", it, eps);
        if (eps < MAXEPS) break;
    }

    endt = omp_get_wtime();

    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", nx, ny, nz);
    printf(" Iterations      =       %12d\n", ITMAX);
    printf(" Time in seconds =       %12.2f\n", endt - startt);
    printf(" Operation type  =     double\n");
    printf(" Threads used    =       %12d\n", omp_get_max_threads());
    printf(" END OF ADI Benchmark\n");

    free(a);
    return 0;
}