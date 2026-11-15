#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

// Error checking macro for CUDA API calls
#define CHECK(action) \
    do { \
        cudaError_t err = (action); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_LAUNCH() CHECK(cudaGetLastError())

#define PI 3.14159265358979323846

enum { kBlockSize = 256, kStreams = 8, kReportCount = 3 };

// define structure for Point:
struct PointParams {
    int id;
    double x1, x2, a, b, c;
};

// define structure for Point Coordinates:
struct PointCoords {
    double x, y;
};

// Calculate the squared distance between two points
__device__ double calculateDistanceSq(PointCoords p1, PointCoords p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx*dx + dy*dy;
}

// Calculate the coordinates of a point given its parameters and a time t
__global__ void calcCoordsKernel(PointParams* d_params, PointCoords* d_coords, int N, double t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Fetch the parameters for the current point
    PointParams p = d_params[idx];
   
    // Calculate X using the formula from the project description
    double x = ((p.x2 - p.x1) / 2.0) * sin(t * PI / 2.0) + (p.x2 + p.x1) / 2.0;
    // Calculate Y using the quadratic formula
    double y = p.a * (x * x) + p.b * x + p.c;
    
    // Store the calculated coordinates in the output array
    d_coords[idx].x = x;
    d_coords[idx].y = y;
}

// One thread per point. Coordinates are walked in shared-memory tiles.
// Threads must not return early: the whole block has to reach both __syncthreads calls.
// Once kReportCount points are recorded, the block stops scanning but still participates in the barriers.
__global__ void checkProximityKernel(PointCoords* d_coords, int* d_ids, int N, int K, double D, int* d_foundPoints, int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = idx < N;

    PointCoords myCoords;
    myCoords.x = 0.0;
    myCoords.y = 0.0;
    if (active) {
        myCoords = d_coords[idx];
    }

    __shared__ double tileX[kBlockSize];
    __shared__ double tileY[kBlockSize];
    __shared__ int stop;

    int neighborsCount = 0;
    bool recorded = false;
    double DSq = D * D;

    for (int base = 0; base < N; base += kBlockSize) {
        if (threadIdx.x == 0) {
            stop = atomicAdd(d_count, 0) >= kReportCount;
        }

        int loadIdx = base + threadIdx.x;
        if (loadIdx < N) {
            tileX[threadIdx.x] = d_coords[loadIdx].x;
            tileY[threadIdx.x] = d_coords[loadIdx].y;
        }
        __syncthreads();

        int remaining = N - base;
        int tileCount = remaining < kBlockSize ? remaining : kBlockSize;
        if (active && !recorded && !stop) {
            for (int k = 0; k < tileCount; k++) {
                int j = base + k;
                if (j == idx) continue;

                PointCoords other;
                other.x = tileX[k];
                other.y = tileY[k];
                if (calculateDistanceSq(myCoords, other) <= DSq) {
                    neighborsCount++;
                }
                if (neighborsCount >= K) {
                    int pos = atomicAdd(d_count, 1);
                    if (pos < kReportCount) {
                        d_foundPoints[pos] = d_ids[idx];
                    }
                    recorded = true;
                    break;
                }
            }
        }
        __syncthreads();
    }
}

// Returns how many sampled times produced at least kReportCount qualifying points.
int runProximityAnalysis(PointParams* h_params, int N, int K, double D, int tCount) {
    PointParams *d_params;
    PointCoords *d_coords_array; // Global memory for coordinates (per stream)
    int *d_ids;
    int *d_foundPoints_array; // Streamed array to store found points for each stream
    int *d_counts_array;      // Streamed array to store counts for each stream
    int timestampsFound = 0;
    // CUDA Events for timing measurement
    float milliseconds = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    // Open output file for writing results
    FILE *fp = fopen("output.txt", "w");
    if (fp == NULL) {
        printf("Error: Could not open output.txt for writing.\n");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }

    int *h_ids = (int*)malloc(N * sizeof(int));
    if (h_ids == NULL) {
        printf("Error: Memory allocation failed for point ids\n");
        fclose(fp);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1;
    }
    for (int i = 0; i < N; i++) h_ids[i] = h_params[i].id;

    // --- Memory Allocation ---
    CHECK(cudaMalloc(&d_params, N * sizeof(PointParams)));
    CHECK(cudaMalloc(&d_ids, N * sizeof(int)));
    // One coordinate buffer per stream so in-flight t values do not overwrite each other
    CHECK(cudaMalloc(&d_coords_array, kStreams * N * sizeof(PointCoords)));
    CHECK(cudaMalloc(&d_foundPoints_array, kStreams * kReportCount * sizeof(int)));
    CHECK(cudaMalloc(&d_counts_array, kStreams * sizeof(int)));
    
    // Copy point parameters to GPU 
    CHECK(cudaMemcpy(d_params, h_params, N * sizeof(PointParams), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ids, h_ids, N * sizeof(int), cudaMemcpyHostToDevice));

    cudaStream_t streams[kStreams];
    for (int s = 0; s < kStreams; s++) {
        CHECK(cudaStreamCreate(&streams[s]));
    }

    printf("Executing parallel proximity check with %d streams...\n", kStreams);

    // Start timing
    CHECK(cudaEventRecord(start));

    // --- Main Computation Loop ---
    for (int i = 0; i <= tCount; i++) {
        int s = i % kStreams;
        double t = (2.0 * i / tCount) - 1.0;

        // Reset the counter for the current stream asynchronously
        CHECK(cudaMemsetAsync(d_counts_array + s, 0, sizeof(int), streams[s]));

        int blocksPerGrid = (N + kBlockSize - 1) / kBlockSize;

        // Step 1: Calculate (x, y) coordinates for all points at time 't'
        // Results are stored in the global memory segment dedicated to this stream
        calcCoordsKernel<<<blocksPerGrid, kBlockSize, 0, streams[s]>>>(
            d_params, d_coords_array + (s * N), N, t
        );
        CHECK_LAUNCH();

        // Step 2: Check Proximity Criteria using the pre-calculated coordinates
        checkProximityKernel<<<blocksPerGrid, kBlockSize, 0, streams[s]>>>(
            d_coords_array + (s * N), d_ids, N, K, D,
            d_foundPoints_array + (s * kReportCount), d_counts_array + s
        );
        CHECK_LAUNCH();

        // Synchronize and write results after every batch of streams or at the end
        if (i % kStreams == kStreams - 1 || i == tCount) {
            for (int j = 0; j < kStreams; j++) {
                int t_idx = i - (i % kStreams) + j;
                if (t_idx > tCount) break;

                // Ensure the stream has finished its kernels
                CHECK(cudaStreamSynchronize(streams[j]));
                
                int h_count;
                CHECK(cudaMemcpy(&h_count, d_counts_array + j, sizeof(int), cudaMemcpyDeviceToHost));

                if (h_count >= kReportCount) {
                    int h_found[kReportCount];
                    CHECK(cudaMemcpy(h_found, d_foundPoints_array + (j * kReportCount), kReportCount * sizeof(int), cudaMemcpyDeviceToHost));

                    double current_t = (2.0 * t_idx / tCount) - 1.0;
                    fprintf(fp, "Points %d, %d, %d satisfy Proximity Criteria at t = %f\n",
                            h_found[0], h_found[1], h_found[2], current_t);
                    timestampsFound++;
                }
            }
        }
    }

    // Stop timing
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Total GPU computation time: %f ms\n", milliseconds);

    // If we never found any 3 points for any t value, we write a message to the output file indicating that.
    if (timestampsFound == 0) {
        fprintf(fp, "There were no 3 points found for any t checked.\n");
    }

    // --- Cleanup ---
    for (int s = 0; s < kStreams; s++) {
        CHECK(cudaStreamDestroy(streams[s]));
    }
    
    fclose(fp);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    CHECK(cudaFree(d_params));
    CHECK(cudaFree(d_ids));
    CHECK(cudaFree(d_coords_array));
    CHECK(cudaFree(d_foundPoints_array));
    CHECK(cudaFree(d_counts_array));
    free(h_ids);
    return timestampsFound;
}

// Same search on one CPU core. Returns how many sampled times produced kReportCount qualifying points.
// The coordinate buffer is allocated once. The scan for a given t stops after kReportCount hits.
int runSequentialAnalysis(PointParams* h_params, int N, int K, double D, int tCount) {
    printf("Starting Sequential (CPU) Analysis for comparison...\n");

    PointCoords* coords = (PointCoords*)malloc(N * sizeof(PointCoords));
    if (coords == NULL) {
        printf("Error: Memory allocation failed for CPU coordinates\n");
        return -1;
    }

    clock_t start = clock();

    int timestampsFound = 0;
    double DSq = D * D;

    for (int i = 0; i <= tCount; i++) {
        double t = (2.0 * i / tCount) - 1.0;

        for (int j = 0; j < N; j++) {
            double x = ((h_params[j].x2 - h_params[j].x1) / 2.0) * sin(t * PI / 2.0) + (h_params[j].x2 + h_params[j].x1) / 2.0;
            double y = h_params[j].a * (x * x) + h_params[j].b * x + h_params[j].c;
            coords[j].x = x;
            coords[j].y = y;
        }

        int pointsMeetingCriteria = 0;
        for (int j = 0; j < N && pointsMeetingCriteria < kReportCount; j++) {
            int neighbors = 0;
            for (int m = 0; m < N; m++) {
                if (j == m) continue;

                double dx = coords[j].x - coords[m].x;
                double dy = coords[j].y - coords[m].y;
                if ((dx * dx + dy * dy) <= DSq) {
                    neighbors++;
                }

                if (neighbors >= K) {
                    pointsMeetingCriteria++;
                    break;
                }
            }
        }

        if (pointsMeetingCriteria >= kReportCount) {
            timestampsFound++;
        }
    }

    clock_t end = clock();
    free(coords);

    double cpu_time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("CPU Execution Time: %.3f ms\n", cpu_time_ms);
    printf("-------------------------------------------\n");
    return timestampsFound;
}

// Main function to read input data, call the GPU analysis function, and handle output
int main(int argc, char* argv[]) {
    int N, K, tCount;
    double D;
    PointParams* h_params;

    // Open the input file for reading
    FILE* inputFile = fopen("input.txt", "r");
    if (inputFile == NULL) {
        printf("Error: Could not open input.txt\n");
        return 1;
    }

    // Read the first line of the input file to get N, K, D, and tCount
    if (fscanf(inputFile, "%d %d %lf %d", &N, &K, &D, &tCount) != 4) {
        printf("Error: Invalid input format in the first line\n");
        fclose(inputFile);
        return 1;
    }

    if (N <= 0 || tCount <= 0) {
        printf("Error: N and tCount must be positive\n");
        fclose(inputFile);
        return 1;
    }

    // Allocate memory for the point parameters on the host
    h_params = (PointParams*)malloc(N * sizeof(PointParams));
    if (h_params == NULL) {
        printf("Error: Memory allocation failed for h_params\n");
        fclose(inputFile);
        return 1;
    }

    // Read the parameters for each point from the input file (ID, x1, x2, a, b, c)
    for (int i = 0; i < N; i++) {
        if (fscanf(inputFile, "%d %lf %lf %lf %lf %lf", 
                   &h_params[i].id, &h_params[i].x1, &h_params[i].x2, 
                   &h_params[i].a, &h_params[i].b, &h_params[i].c) != 6) {
            printf("Error reading point %d data\n", i);
            free(h_params);
            fclose(inputFile);
            return 1;
        }
    }

    // Close the input file after reading all data
    fclose(inputFile);

    printf("Successfully loaded %d points. Starting GPU Analysis...\n", N);
    
    int gpuHits = runProximityAnalysis(h_params, N, K, D, tCount);
    int cpuHits = runSequentialAnalysis(h_params, N, K, D, tCount);
    if (gpuHits < 0 || cpuHits < 0) {
        free(h_params);
        return 1;
    }

    printf("Timestamps with at least %d qualifying points: GPU=%d CPU=%d\n", kReportCount, gpuHits, cpuHits);
    if (gpuHits != cpuHits) {
        printf("Mismatch between GPU and CPU counts.\n");
        free(h_params);
        return 1;
    }

    printf("GPU and CPU agree. Results saved to output.txt\n");

    // Free the host memory allocated for point parameters
    free(h_params);

    return 0;
}
