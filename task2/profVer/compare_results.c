#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LINE 512

double extract_eps(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE];
    double last_eps = -1.0;

    while (fgets(line, MAX_LINE, fp)) {
        if (strstr(line, "EPS")) {
            sscanf(line, " IT = %*d   EPS = %lf", &last_eps);
        }
    }

    fclose(fp);
    return last_eps;
}

double extract_time(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE];
    double time = -1.0;

    while (fgets(line, MAX_LINE, fp)) {
        if (strstr(line, "Time in seconds")) {
            sscanf(line, " Time in seconds = %lf", &time);
        }
    }

    fclose(fp);
    return time;
}

int main() {
    printf("Running CPU version...\n");
    if (system("./adi3d_cpu > cpu_output.txt") != 0) {
        fprintf(stderr, "Error running CPU version\n");
        return 1;
    }

    printf("Running GPU version...\n");
    if (system("./adi3d_gpu > gpu_output.txt") != 0) {
        fprintf(stderr, "Error running GPU version\n");
        return 1;
    }

    double eps_cpu = extract_eps("cpu_output.txt");
    double eps_gpu = extract_eps("gpu_output.txt");
    double time_cpu = extract_time("cpu_output.txt");
    double time_gpu = extract_time("gpu_output.txt");

    printf("\n--- Comparison Report ---\n");
    printf("CPU eps: %.8lf\n", eps_cpu);
    printf("GPU eps: %.8lf\n", eps_gpu);
    printf("CPU time: %.2f sec\n", time_cpu);
    printf("GPU time: %.2f sec\n", time_gpu);
    printf("Speedup (CPU/GPU): %.2fx\n", time_cpu / time_gpu);

    if (fabs(eps_cpu - eps_gpu) < 1e-6) {
        printf("✅ EPS values match (within threshold).\n");
    } else {
        printf("❌ EPS values differ.\n");
    }

    return 0;
}