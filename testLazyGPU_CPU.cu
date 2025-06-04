#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <climits>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>

using namespace std;
using namespace std::chrono;

// Score constants
const int G_INIT = 1;
const int G_EXT = 1;
const int MATCH = 1;
const int MISMATCH = -1;

// --- CPU Lazy ---
int score(unsigned char s1, unsigned char s2) {
    return (s1 == s2) ? MATCH : MISMATCH;
}

int LazySmith(unsigned char *seq1, unsigned char *seq2, int n, int m) {
    std::vector<int> H_prev(n+1, 0), H_curr(n+1, 0);
    std::vector<int> E(n+1, 0);
    std::vector<int> F(n+1, 0);

    int maxScore = 0;

    for (int i = 1; i <= m; ++i) {
        E[0]      = 0;
        F[0]      = 0;
        H_curr[0] = 0;

        for (int j = 1; j <= n; ++j) {
            E[j] = std::max(E[j-1] - G_EXT, H_curr[j-1] - G_INIT);

            int Ht = H_prev[j-1] + score(seq1[j-1], seq2[i-1]);

            F[j] = std::max(F[j] - G_EXT, H_prev[j] - G_INIT);

            int hval = Ht;
            if (E[j] > hval)  hval = E[j];
            if (F[j] > hval)  hval = F[j];
            if (hval < 0)     hval = 0;
            H_curr[j] = hval;

            if (hval > maxScore) maxScore = hval;
        }

        for (int j = 1; j <= n; ++j) {
            int fij = H_curr[j] - G_INIT;
            if (fij > F[j]) {
                F[j] = fij;
                if (F[j] > H_curr[j]) {
                    H_curr[j] = F[j];
                    if (H_curr[j] > maxScore) maxScore = H_curr[j];
                }
                for (int k = j+1; k <= n; ++k) {
                    int newF = F[k-1] - G_EXT;
                    if (newF <= 0) break;
                    if (newF <= F[k]) break;
                    F[k] = newF;
                    if (F[k] > H_curr[k]) {
                        H_curr[k] = F[k];
                        if (H_curr[k] > maxScore) maxScore = H_curr[k];
                    }
                }
            }
        }

        std::swap(H_prev, H_curr);
        std::fill(H_curr.begin(), H_curr.end(), 0);
    }

    return maxScore;
}

// --- GPU Lazy ---
__device__ __forceinline__
int device_score(unsigned char a, unsigned char b) {
    return (a == b) ? MATCH : MISMATCH;
}

__global__
void sw_kernel_diag(int d, int n, int m, const unsigned char* seq1, const unsigned char* seq2, int* H, int* E, int* F) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_i = max(1, d - n);
    int end_i = min(d - 1, m);
    int diag_len = end_i - start_i + 1;
    if (tid >= diag_len) return;

    int i = start_i + tid;
    int j = d - i;

    int idx = i * (n + 1) + j;
    int idx_up = (i - 1) * (n + 1) + j;
    int idx_left = i * (n + 1) + (j - 1);
    int idx_diag = (i - 1) * (n + 1) + (j - 1);

    int e = max(H[idx_left] - G_INIT, E[idx_left] - G_EXT);
    int f = max(H[idx_up] - G_INIT, F[idx_up] - G_EXT);
    int match = H[idx_diag] + device_score(seq1[j - 1], seq2[i - 1]);
    int h_val = max(0, max(match, max(e, f)));

    E[idx] = e;
    F[idx] = f;
    H[idx] = h_val;
}

int SmithWatermanLazyGPU(const unsigned char* seq1, const unsigned char* seq2, int n, int m) {
    size_t matrix_size = (size_t)(m + 1) * (size_t)(n + 1);
    int *dH, *dE, *dF;
    cudaMalloc(&dH, matrix_size * sizeof(int));
    cudaMalloc(&dE, matrix_size * sizeof(int));
    cudaMalloc(&dF, matrix_size * sizeof(int));
    cudaMemset(dH, 0, matrix_size * sizeof(int));
    cudaMemset(dE, 0, matrix_size * sizeof(int));
    cudaMemset(dF, 0, matrix_size * sizeof(int));

    unsigned char *d_seq1, *d_seq2;
    cudaMalloc(&d_seq1, n * sizeof(unsigned char));
    cudaMalloc(&d_seq2, m * sizeof(unsigned char));
    cudaMemcpy(d_seq1, seq1, n * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq2, seq2, m * sizeof(unsigned char), cudaMemcpyHostToDevice);

    const int THREADS = 256;
    for (int d = 2; d <= n + m; ++d) {
        int start_i = std::max(1, d - n);
        int end_i   = std::min(d - 1, m);
        int diag_len = (end_i >= start_i) ? (end_i - start_i + 1) : 0;
        if (diag_len <= 0) continue;
        int blocks = (diag_len + THREADS - 1) / THREADS;
        sw_kernel_diag<<<blocks, THREADS>>>(d, n, m, d_seq1, d_seq2, dH, dE, dF);
        cudaDeviceSynchronize();
    }

    vector<int> h_H(matrix_size);
    cudaMemcpy(h_H.data(), dH, matrix_size * sizeof(int), cudaMemcpyDeviceToHost);

    int best = 0;
    for (int val : h_H) best = max(best, val);

    cudaFree(d_seq1); cudaFree(d_seq2);
    cudaFree(dH); cudaFree(dE); cudaFree(dF);
    return best;
}

// --- GPU Classic ---
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

    int e_val = max(dE[idx_i_jm1] - G_EXT, dH[idx_i_jm1] - G_INIT);
    int f_val = max(dF[idx_im1_j] - G_EXT, dH[idx_im1_j] - G_INIT);
    int diag_score = dH[idx_im1_jm1] + device_score(d_q[j - 1], d_d[i - 1]);

    int mm_val = max(0, diag_score);
    int gap_val = max(e_val, f_val);
    int h_val = max(mm_val, gap_val);
    dH[idx] = h_val;
    dE[idx] = e_val;
    dF[idx] = f_val;
}

int SmithWatermanScoreCUDA(const unsigned char* seq1, const unsigned char* seq2, int n, int m) {
    size_t matrix_size = (size_t)(m + 1) * (size_t)(n + 1);
    std::vector<int> h_H(matrix_size, 0);
    int *dH, *dE, *dF;
    cudaMalloc(&dH, matrix_size * sizeof(int));
    cudaMalloc(&dE, matrix_size * sizeof(int));
    cudaMalloc(&dF, matrix_size * sizeof(int));
    cudaMemset(dH, 0, matrix_size * sizeof(int));
    cudaMemset(dE, 0, matrix_size * sizeof(int));
    cudaMemset(dF, 0, matrix_size * sizeof(int));

    unsigned char *d_seq1, *d_seq2;
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
            dE, dF, dH, d_seq1, d_seq2, n, m, k
        );
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_H.data(), dH, matrix_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_seq1); cudaFree(d_seq2);
    cudaFree(dE); cudaFree(dF); cudaFree(dH);

    int best = 0;
    for (int v : h_H) best = max(best, v);
    return best;
}

// --- Main ---
int main() {
    vector<pair<string, string>> tests;
    mt19937_64 rng(42);
    uniform_int_distribution<int> dist(0, 3);
    const char* nts = "ACGT";
    for (int i = 0; i < 20; ++i) {
        string s1, s2;
        int len = 5000 + i * 1000;
        for (int j = 0; j < len; ++j) {
            s1 += nts[dist(rng)];
            s2 += nts[dist(rng)];
        }
        tests.emplace_back(move(s1), move(s2));
    }

    for (size_t i = 0; i < tests.size(); ++i) {
        const auto& [s1, s2] = tests[i];
        vector<unsigned char> seq1(s1.begin(), s1.end()), seq2(s2.begin(), s2.end());
        int n = seq1.size(), m = seq2.size();

        auto t1 = high_resolution_clock::now();
        int score_cpu = LazySmith(seq1.data(), seq2.data(), n, m);
        auto t2 = high_resolution_clock::now();
        int score_gpu_lazy = SmithWatermanLazyGPU(seq1.data(), seq2.data(), n, m);
        auto t3 = high_resolution_clock::now();
        int score_gpu_classic = SmithWatermanScoreCUDA(seq1.data(), seq2.data(), n, m);
        auto t4 = high_resolution_clock::now();

        auto cpu_time = duration_cast<milliseconds>(t2 - t1).count();
        auto gpu_lazy_time = duration_cast<milliseconds>(t3 - t2).count();
        auto gpu_classic_time = duration_cast<milliseconds>(t4 - t3).count();

        cout << "Test " << i + 1 << " (" << n << "x" << m << ")\n";
        cout << "  CPU Lazy       = " << score_cpu << ", Time = " << cpu_time << " ms\n";
        cout << "  GPU Lazy       = " << score_gpu_lazy << ", Time = " << gpu_lazy_time << " ms\n";
        cout << "  GPU Classic    = " << score_gpu_classic << ", Time = " << gpu_classic_time << " ms\n";
        if (score_cpu != score_gpu_classic || score_gpu_lazy != score_gpu_classic)
            cout << "  ⚠️ Score mismatch detected\n";
        cout << endl;
    }

    return 0;
}


/*
LLMs were used to put together the tests function (the algorithms are
from the other files).
*/