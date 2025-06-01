#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cassert>

static const int G_INIT_CPU   = 1;   // penalty for opening a gap
static const int G_EXT_CPU    = 1;   // penalty for extending a gap
static const int MATCH_CPU    = 1;   // score for a match
static const int MISMATCH_CPU = -1;  // score for a mismatch

inline int score_cpu(unsigned char a, unsigned char b) {
    return (a == b) ? MATCH_CPU : MISMATCH_CPU;
}

void dpMatricesCPU(
    std::vector<std::vector<int>>& E,
    std::vector<std::vector<int>>& F,
    std::vector<std::vector<int>>& H,
    const unsigned char* q,
    const unsigned char* d,
    int n,
    int m
) {
    for (int i = 0; i <= m; ++i) {
        E[i][0] = 0;
        F[i][0] = 0;
        H[i][0] = 0;
    }
    for (int j = 0; j <= n; ++j) {
        E[0][j] = 0;
        F[0][j] = 0;
        H[0][j] = 0;
    }

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            E[i][j] = std::max(E[i][j - 1] - G_EXT_CPU, H[i][j - 1] - G_INIT_CPU);
            F[i][j] = std::max(F[i - 1][j] - G_EXT_CPU, H[i - 1][j] - G_INIT_CPU);

            int gap_val = std::max(E[i][j], F[i][j]);
            int mm_val  = H[i - 1][j - 1] + score_cpu(q[j - 1], d[i - 1]);
            H[i][j] = std::max(0, std::max(gap_val, mm_val));
        }
    }
}

int SmithWatermanScoreCPU(
    const unsigned char* seq1,
    const unsigned char* seq2,
    int n,
    int m
) {
    std::vector<std::vector<int>> E(m + 1, std::vector<int>(n + 1, 0));
    std::vector<std::vector<int>> F(m + 1, std::vector<int>(n + 1, 0));
    std::vector<std::vector<int>> H(m + 1, std::vector<int>(n + 1, 0));

    dpMatricesCPU(E, F, H, seq1, seq2, n, m);

    int best = 0;
    for (int i = 0; i <= m; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (H[i][j] > best) best = H[i][j];
        }
    }
    return best;
}


// ------------------------------------------------------------
//                        CUDA version
// ------------------------------------------------------------

static const int G_INIT_CUDA   = 1;
static const int G_EXT_CUDA    = 1;
static const int MATCH_CUDA    = 1;
static const int MISMATCH_CUDA = -1;

__device__ __forceinline__
int device_score(unsigned char a, unsigned char b) {
    return (a == b) ? MATCH_CUDA : MISMATCH_CUDA;
}

__global__
void kernel_compute_diagonal(
    int* __restrict__ dE,
    int* __restrict__ dF,
    int* __restrict__ dH,
    const unsigned char* __restrict__ d_q,
    const unsigned char* __restrict__ d_d,
    int n,
    int m,
    int k
) {
    int i_min = max(1, k - n);
    int i_max = min(m, k - 1);
    int diag_len = i_max - i_min + 1;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= diag_len) return;

    int i = i_min + t;
    int j = k - i; 

    int idx       = i * (n + 1) + j;
    int idx_im1_j = (i - 1) * (n + 1) + j;
    int idx_i_jm1 = i * (n + 1) + (j - 1);
    int idx_im1_jm1 = (i - 1) * (n + 1) + (j - 1);

    int e_val = max(dE[idx_i_jm1] - G_EXT_CUDA,
                    dH[idx_i_jm1] - G_INIT_CUDA);
    dE[idx] = e_val;

    int f_val = max(dF[idx_im1_j] - G_EXT_CUDA,
                    dH[idx_im1_j] - G_INIT_CUDA);
    dF[idx] = f_val;

    int diag_score = dH[idx_im1_jm1] + device_score(d_q[j - 1], d_d[i - 1]);
    int mm_val = max(0, diag_score);
    int gap_val = max(e_val, f_val);
    int h_val = max(mm_val, gap_val);
    dH[idx] = h_val;
}

int SmithWatermanScoreCUDA(
    const unsigned char* seq1,
    const unsigned char* seq2,
    int n,
    int m
) {
    size_t matrix_size = (size_t)(m + 1) * (size_t)(n + 1);

    std::vector<int> h_H(matrix_size, 0);

    int *dH = nullptr, *dE = nullptr, *dF = nullptr;
    cudaMalloc(&dH, matrix_size * sizeof(int));
    cudaMalloc(&dE, matrix_size * sizeof(int));
    cudaMalloc(&dF, matrix_size * sizeof(int));

    cudaMemset(dH, 0, matrix_size * sizeof(int));
    cudaMemset(dE, 0, matrix_size * sizeof(int));
    cudaMemset(dF, 0, matrix_size * sizeof(int));

    unsigned char *d_seq1 = nullptr, *d_seq2 = nullptr;
    cudaMalloc(&d_seq1, n * sizeof(unsigned char));
    cudaMalloc(&d_seq2, m * sizeof(unsigned char));
    cudaMemcpy(d_seq1, seq1, n * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq2, seq2, m * sizeof(unsigned char), cudaMemcpyHostToDevice);

    const int THREADS_PER_BLOCK = 256;

    for (int k = 2; k <= m + n; ++k) {
        int i_min = std::max(1, k - n);
        int i_max = std::min(m, k - 1);
        int diag_len = (i_max >= i_min) ? (i_max - i_min + 1) : 0;
        if (diag_len <= 0) continue;
        int blocks = (diag_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        kernel_compute_diagonal<<<blocks, THREADS_PER_BLOCK>>>(
            dE, dF, dH,
            d_seq1, d_seq2,
            n, m, k
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed at k=" << k
                      << ": " << cudaGetErrorString(err) << std::endl;
            break;
        }
    }

    cudaMemcpy(h_H.data(), dH, matrix_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_seq1);
    cudaFree(d_seq2);
    cudaFree(dE);
    cudaFree(dF);
    cudaFree(dH);

    int best = 0;
    for (size_t idx = 0; idx < h_H.size(); ++idx) {
        if (h_H[idx] > best) best = h_H[idx];
    }
    return best;
}


// ------------------------------------------------------------
//                           Main / Timing
// ------------------------------------------------------------

int main() {
    const int NUM_EXAMPLES = 10;
    const int SEQ_LENGTH   = 4096; 

    std::mt19937_64 rng(12345); 
    std::uniform_int_distribution<int> nucleo_dist(0, 3);
    const char nucleotides[4] = {'A', 'C', 'G', 'T'};

    std::vector<std::vector<unsigned char>> seq1_list(NUM_EXAMPLES),
                                            seq2_list(NUM_EXAMPLES);
    for (int ex = 0; ex < NUM_EXAMPLES; ++ex) {
        seq1_list[ex].resize(SEQ_LENGTH);
        seq2_list[ex].resize(SEQ_LENGTH);
        for (int i = 0; i < SEQ_LENGTH; ++i) {
            seq1_list[ex][i] = nucleotides[nucleo_dist(rng)];
            seq2_list[ex][i] = nucleotides[nucleo_dist(rng)];
        }
    }

    std::cout << "Running " << NUM_EXAMPLES
              << " examples (sequences of length " << SEQ_LENGTH << ").\n\n";

    std::cout << "=== CPU (sequential) timings ===\n";
    for (int ex = 0; ex < NUM_EXAMPLES; ++ex) {
        const unsigned char* s1 = seq1_list[ex].data();
        const unsigned char* s2 = seq2_list[ex].data();
        int n = SEQ_LENGTH;
        int m = SEQ_LENGTH;

        auto t0 = std::chrono::high_resolution_clock::now();
        int cpu_score = SmithWatermanScoreCPU(s1, s2, n, m);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "Example " << (ex + 1)
                  << ": Score = " << cpu_score
                  << ", Time = " << cpu_ms << " ms\n";
    }
    std::cout << "\n";

    std::cout << "=== CUDA (GPU) timings ===\n";

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    for (int ex = 0; ex < NUM_EXAMPLES; ++ex) {
        const unsigned char* s1 = seq1_list[ex].data();
        const unsigned char* s2 = seq2_list[ex].data();
        int n = SEQ_LENGTH;
        int m = SEQ_LENGTH;

        cudaEventRecord(start_event, 0);
        int gpu_score = SmithWatermanScoreCUDA(s1, s2, n, m);
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);

        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, start_event, stop_event);

        std::cout << "Example " << (ex + 1)
                  << ": Score = " << gpu_score
                  << ", Time = " << gpu_ms << " ms\n";

        int cpu_check = SmithWatermanScoreCPU(s1, s2, n, m);
        if (cpu_check != gpu_score) {
            std::cerr << "  [Warning] Mismatch on example " << (ex + 1)
                      << ": CPU=" << cpu_check << ", GPU=" << gpu_score << "\n";
        }
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}


/*
LLMs were used for the code handling timing.
*/

/*
Results:

===================================================================
Short sequences: 32

=== CPU (sequential) timings ===
Example 1: Score = 6, Time = 0.016527 ms
Example 2: Score = 16, Time = 0.008599 ms
Example 3: Score = 5, Time = 0.008155 ms
Example 4: Score = 6, Time = 0.007873 ms
Example 5: Score = 5, Time = 0.007887 ms
Example 6: Score = 6, Time = 0.007754 ms
Example 7: Score = 6, Time = 0.007695 ms
Example 8: Score = 8, Time = 0.007659 ms
Example 9: Score = 8, Time = 0.007749 ms
Example 10: Score = 5, Time = 0.007651 ms

=== CUDA (GPU) timings ===
Example 1: Score = 6, Time = 5.22307 ms
Example 2: Score = 16, Time = 0.214016 ms
Example 3: Score = 5, Time = 0.197376 ms
Example 4: Score = 6, Time = 0.188128 ms
Example 5: Score = 5, Time = 0.191552 ms
Example 6: Score = 6, Time = 0.183584 ms
Example 7: Score = 6, Time = 0.189664 ms
Example 8: Score = 8, Time = 0.18992 ms
Example 9: Score = 8, Time = 0.191936 ms
Example 10: Score = 5, Time = 0.190944 ms

===================================================================

===================================================================
Medium sequences: 516

=== CPU (sequential) timings ===
Example 1: Score = 65, Time = 2.11451 ms
Example 2: Score = 56, Time = 1.35076 ms
Example 3: Score = 56, Time = 1.29297 ms
Example 4: Score = 72, Time = 1.32204 ms
Example 5: Score = 69, Time = 1.30111 ms
Example 6: Score = 61, Time = 1.2838 ms
Example 7: Score = 70, Time = 1.28541 ms
Example 8: Score = 62, Time = 1.31768 ms
Example 9: Score = 56, Time = 1.29069 ms
Example 10: Score = 56, Time = 1.2802 ms

=== CUDA (GPU) timings ===
Example 1: Score = 65, Time = 6.95536 ms
Example 2: Score = 56, Time = 2.89526 ms
Example 3: Score = 56, Time = 2.87437 ms
Example 4: Score = 72, Time = 2.85869 ms
Example 5: Score = 69, Time = 2.85814 ms
Example 6: Score = 61, Time = 2.84432 ms
Example 7: Score = 70, Time = 2.84611 ms
Example 8: Score = 62, Time = 2.84752 ms
Example 9: Score = 56, Time = 2.85168 ms
Example 10: Score = 56, Time = 2.84371 ms

===================================================================
===================================================================
Long sequences: 4096

=== CPU (sequential) timings ===
Example 1: Score = 466, Time = 123.608 ms
Example 2: Score = 472, Time = 108.316 ms
Example 3: Score = 474, Time = 107.097 ms
Example 4: Score = 497, Time = 107.004 ms
Example 5: Score = 486, Time = 105.14 ms
Example 6: Score = 442, Time = 104.956 ms
Example 7: Score = 454, Time = 105.003 ms
Example 8: Score = 478, Time = 105.714 ms
Example 9: Score = 451, Time = 105.689 ms
Example 10: Score = 471, Time = 106.472 ms

=== CUDA (GPU) timings ===
Example 1: Score = 466, Time = 48.1953 ms
Example 2: Score = 472, Time = 45.167 ms
Example 3: Score = 474, Time = 44.9787 ms
Example 4: Score = 497, Time = 45.2071 ms
Example 5: Score = 486, Time = 45.1173 ms
Example 6: Score = 442, Time = 45.4459 ms
Example 7: Score = 454, Time = 45.397 ms
Example 8: Score = 478, Time = 45.2494 ms
Example 9: Score = 451, Time = 43.6426 ms
Example 10: Score = 471, Time = 43.9214 ms

*/