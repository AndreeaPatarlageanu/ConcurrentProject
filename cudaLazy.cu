#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>

using namespace std::chrono;

static const int G_INIT   = 1;
static const int G_EXT    = 1;
static const int MATCH    = 1;
static const int MISMATCH = -1;

__device__ __forceinline__
int device_score(unsigned char a, unsigned char b) {
    return (a == b) ? MATCH : MISMATCH;
}

__global__
void sw_kernel_diag(
    int d,
    int n,
    int m,
    const unsigned char* __restrict__ seq1,
    const unsigned char* __restrict__ seq2,
    int* __restrict__ H,
    int* __restrict__ E,
    int* __restrict__ F
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_i = max(1, d - n);
    int end_i   = min(d - 1, m);
    int diag_len = end_i - start_i + 1;

    if (tid >= diag_len) return;

    int i = start_i + tid;
    int j = d - i;

    int idx      = i * (n + 1) + j;
    int idx_up   = (i - 1) * (n + 1) + j;
    int idx_left = i * (n + 1) + (j - 1);
    int idx_diag = (i - 1) * (n + 1) + (j - 1);

    int e = max(H[idx_left] - G_INIT, E[idx_left] - G_EXT);
    int f = max(H[idx_up]   - G_INIT, F[idx_up]   - G_EXT);
    int match = H[idx_diag] + device_score(seq1[j - 1], seq2[i - 1]);

    int h_val = max(0, max(match, max(e, f)));

    E[idx] = e;
    F[idx] = f;
    H[idx] = h_val;
}

extern "C" int SmithWatermanLazyGPU(const unsigned char* seq1, const unsigned char* seq2, int n, int m) {
    size_t matrix_size = (size_t)(m + 1) * (size_t)(n + 1);

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

    std::vector<int> h_H(matrix_size);
    cudaMemcpy(h_H.data(), dH, matrix_size * sizeof(int), cudaMemcpyDeviceToHost);

    int best = 0;
    for (int val : h_H) best = std::max(best, val);

    cudaFree(d_seq1);
    cudaFree(d_seq2);
    cudaFree(dH);
    cudaFree(dE);
    cudaFree(dF);

    return best;
}

void runLazy() {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(0, 3);
    const char* nts = "ACGT";

    std::vector<std::pair<std::string, std::string>> tests = {
        {"AABDADB", "AADCBAB"},
        {"AAA", "AAA"},
        {std::string(5000, 'A'), std::string(5000, 'A')},
        {std::string(2000, 'A'), std::string(2000, 'T')},
    };

    for (int i = 0; i < 5; ++i) {
        std::string s1, s2;
        for (int j = 0; j < 4000 + i * 1000; ++j) {
            s1 += nts[dist(rng)];
            s2 += nts[dist(rng)];
        }
        tests.emplace_back(std::move(s1), std::move(s2));
    }

    int correct = 0;
    for (size_t i = 0; i < tests.size(); ++i) {
        const auto& tc = tests[i];
        int n = (int)tc.first.size();
        int m = (int)tc.second.size();

        std::vector<unsigned char> h_seq1(n), h_seq2(m);
        for (int j = 0; j < n; ++j) h_seq1[j] = (unsigned char)tc.first[j];
        for (int j = 0; j < m; ++j) h_seq2[j] = (unsigned char)tc.second[j];

        int lazy = SmithWatermanLazyGPU(h_seq1.data(), h_seq2.data(), n, m);

        std::cout << "Test " << i+1 << " (" << n << "x" << m << ")\n";
        std::cout << "  Lazy GPU  = " << lazy << "\n";
    }

}
 

// int main() {
//     runLazy();
//     return 0;
// }