#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>

// Scoring parameters
#define MATCH 1
#define MISMATCH -1
#define G_INIT 1    
#define G_EXT 1     

typedef int ScoreType;

int referenceSmithWaterman(const std::string& a, const std::string& b) {
    int n = a.size(), m = b.size(), best = 0;
    std::vector<std::vector<int>> H(m+1, std::vector<int>(n+1, 0));
    std::vector<std::vector<int>> E(m+1, std::vector<int>(n+1, 0));
    std::vector<std::vector<int>> F(m+1, std::vector<int>(n+1, 0));
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int sc = (a[j-1] == b[i-1] ? MATCH : MISMATCH);
            int diag = H[i-1][j-1] + sc;
            E[i][j] = std::max(H[i][j-1] - G_INIT, E[i][j-1] - G_EXT);
            F[i][j] = std::max(H[i-1][j] - G_INIT, F[i-1][j] - G_EXT);
            int h = std::max(0, std::max(diag, std::max(E[i][j], F[i][j])));
            H[i][j] = h;
            best = std::max(best, h);
        }
    }
    return best;
}

__global__ void smithWatermanDiagonal(
    const char* seq1, const char* seq2,
    ScoreType* H, int cols,
    int diag, int len1, int len2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i_start = max(1, diag - len2);
    int i_end   = min(len1, diag - 1);
    int count   = i_end - i_start + 1;
    if (tid >= count) return;

    int i = i_start + tid;
    int j = diag - i;

    int idx    = i * cols + j;
    int idx_nw = (i - 1) * cols + (j - 1);
    int idx_n  = (i - 1) * cols + j;
    int idx_w  = i * cols + (j - 1);

    int score_sub = (seq1[i - 1] == seq2[j - 1]) ? MATCH : MISMATCH;

    ScoreType val_nw = H[idx_nw] + score_sub;
    ScoreType val_n  = H[idx_n]  - G_INIT;
    ScoreType val_w  = H[idx_w]  - G_INIT;

    ScoreType best = max(0, max(val_nw, max(val_n, val_w)));
    H[idx] = best;
}

std::string randomSequence(int length) {
    static const char bases[] = {'A','C','G','T'};
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<int> dist(0,3);
    std::string s; s.reserve(length);
    for (int i = 0; i < length; ++i)
        s.push_back(bases[dist(rng)]);
    return s;
}

bool runTest(const std::string& s1, const std::string& s2, int testId) {
    std::cout << "  Test " << testId << ": ";

    auto t0 = std::chrono::high_resolution_clock::now();
    int cpuScore = referenceSmithWaterman(s1, s2);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int len1 = s1.size(), len2 = s2.size();
    int rows = len1 + 1, cols = len2 + 1;
    size_t sz = rows * cols * sizeof(ScoreType);
    ScoreType* h_H = (ScoreType*)malloc(sz);
    memset(h_H, 0, sz);

    char *d_s1, *d_s2; ScoreType *d_H;
    cudaMalloc(&d_s1, len1);
    cudaMalloc(&d_s2, len2);
    cudaMalloc(&d_H, sz);
    cudaMemcpy(d_s1, s1.data(), len1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s2, s2.data(), len2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, h_H, sz, cudaMemcpyHostToDevice);

    // GPU
    auto t2 = std::chrono::high_resolution_clock::now();
    const int TPB = 256;
    int maxDiag = len1 + len2;
    for (int diag = 2; diag <= maxDiag; ++diag) {
        int i0 = max(1, diag - len2), i1 = min(len1, diag - 1);
        int count = max(0, i1 - i0 + 1);
        if (!count) continue;
        int blocks = (count + TPB - 1) / TPB;
        smithWatermanDiagonal<<<blocks,TPB>>>(d_s1, d_s2, d_H, cols, diag, len1, len2);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_H, d_H, sz, cudaMemcpyDeviceToHost);
    int gpuScore = 0;
    for (int i = 1; i <= len1; ++i)
        for (int j = 1; j <= len2; ++j)
            gpuScore = std::max(gpuScore, h_H[i*cols + j]);

    free(h_H);
    cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_H);

    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout
      << "CPU score=" << cpuScore
      << ", GPU score=" << gpuScore
      << (cpuScore==gpuScore ? " (match)" : " (mismatch)")
      << ", CPU time=" << cpu_ms << " ms"
      << ", GPU time=" << gpu_ms << " ms"
      << ", speedup=" << (cpu_ms/gpu_ms) << "x"
      << std::endl;

    return cpuScore == gpuScore;
}

int main() {
    const int numTests = 5;  // you can adjust per-length trial count
    const std::vector<int> lengths = {2000,4000,6000,8000,10000,
                                      12000,14000,16000,18000};

    for (int L : lengths) {
        std::cout << "=== Running " << numTests
                  << " tests on random " << L << "-mer sequences ==="
                  << std::endl;
        int passed = 0;
        for (int t = 1; t <= numTests; ++t) {
            auto s1 = randomSequence(L);
            auto s2 = randomSequence(L);
            if (runTest(s1, s2, t)) ++passed;
        }
        std::cout << "Summary for length " << L << ": "
                  << passed << "/" << numTests << " passed."
                  << std::endl << std::endl;
    }
    return 0;
}
