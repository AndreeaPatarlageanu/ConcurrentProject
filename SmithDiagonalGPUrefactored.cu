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
using namespace std::chrono ;

// Scoring parameters
#define MATCH 1
#define MISMATCH -1
#define G_INIT 1    // gap open penalty
#define G_EXT 1     // gap extension penalty

#define TILE_SIZE 32

/*#define nucleotides "ACGT"
static std :: default_random_engine engine(8) ; 
int U(int N) {
    std::uniform_int_distribution<int> dist(0, N - 1);
    return dist(engine);
}
unsigned char randomCharacter() {
    static bool seeded = false;
    if (!seeded) {
        std::srand(std::time(0));
        seeded = true;
    }
    char randomChar = nucleotides[std::rand() % (sizeof(nucleotides) - 1)] ;
    return randomChar ;
}
void randomSequence(int N, unsigned char* seq ) {
    for( int i = 0; i < N; i++ ) {
        seq[i] = randomCharacter() ;
    }
}*/

/*__global__
void warmup() {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
}*/

typedef int ScoreType;

// int referenceSmithWaterman(const std::string& a, const std::string& b) {
int referenceSmithWaterman(const unsigned char *a, const unsigned char *b, int n, int m) {

    std::vector<std::vector<int>> H(m + 1, std::vector<int>(n + 1, 0));
    std::vector<std::vector<int>> E(m + 1, std::vector<int>(n + 1, 0));
    std::vector<std::vector<int>> F(m + 1, std::vector<int>(n + 1, 0));
    int best = 0;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int sc = (a[j - 1] == b[i - 1] ? MATCH : MISMATCH);
            int diag = H[i - 1][j - 1] + sc;
            E[i][j] = std::max(H[i][j - 1] - G_INIT, E[i][j - 1] - G_EXT);
            F[i][j] = std::max(H[i - 1][j] - G_INIT, F[i - 1][j] - G_EXT);
            int h = std::max(0, std::max(diag, std::max(E[i][j], F[i][j])));
            H[i][j] = h;
            best = std::max(best, h);
        }
    }
    return best;
}

__global__ void smithWatermanDiagonal(
    const unsigned char* seq1, const unsigned char* seq2,
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

// Generates a random DNA sequence of given length
std::string randomSequence(int length) {
    static const char bases[] = {'A', 'C', 'G', 'T'};
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<int> dist(0, 3);
    std::string s; s.reserve(length);
    for (int i = 0; i < length; ++i) s.push_back(bases[dist(rng)]);
    return s;
}

/*bool runTest(const std::string& seq1_str, const std::string& seq2_str, int testId) {
    std::cout << "Test " << testId << ": ";
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int cpuScore = referenceSmithWaterman(seq1_str, seq2_str);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    int len1 = seq1_str.size();
    int len2 = seq2_str.size();
    int rows = len1 + 1;
    int cols = len2 + 1;
    size_t size = rows * cols * sizeof(ScoreType);

    ScoreType* h_H = (ScoreType*)malloc(size);
    memset(h_H, 0, size);
    char* d_seq1; char* d_seq2; ScoreType* d_H;
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_seq1, len1);
    cudaMalloc(&d_seq2, len2);
    cudaMalloc(&d_H, size);
    cudaMemcpy(d_seq1, seq1_str.data(), len1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq2, seq2_str.data(), len2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, h_H, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int maxDiag = len1 + len2;
    for (int diag = 2; diag <= maxDiag; ++diag) {
        int i_start = max(1, diag - len2);
        int i_end   = min(len1, diag - 1);
        int count   = max(0, i_end - i_start + 1);
        if (count <= 0) continue;
        int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
        smithWatermanDiagonal<<<blocks, threadsPerBlock>>>(
            d_seq1, d_seq2, d_H, cols, diag, len1, len2);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_H, d_H, size, cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    int gpuScore = 0;
    for (int i = 1; i <= len1; ++i)
        for (int j = 1; j <= len2; ++j)
            gpuScore = std::max(gpuScore, h_H[i * cols + j]);

    // Cleanup
    free(h_H);
    cudaFree(d_seq1);
    cudaFree(d_seq2);
    cudaFree(d_H);

    std::cout << "CPU score=" << cpuScore << ", GPU score=" << gpuScore;
    if (cpuScore == gpuScore) std::cout << " (match)";
    else std::cout << " (mismatch)";
    std::cout << ", CPU time=" << cpu_ms << " ms"
              << ", GPU time=" << gpu_ms << " ms"
              << ", speedup=" << (cpu_ms / gpu_ms) << "x"
              << std::endl;

    return cpuScore == gpuScore;
}*/

extern "C" int SmithDiagonalGPU(unsigned char *seq1, unsigned char *seq2, int n, int m) {

    int rows = n + 1;
    int cols = m + 1;
    size_t size = rows * cols * sizeof(ScoreType);
    size_t s = sizeof(unsigned char) ;

    ScoreType* h_H = (ScoreType*)malloc(size);
    memset(h_H, 0, size);
    unsigned char* d_seq1; 
    unsigned char* d_seq2; 
    ScoreType* d_H;

    cudaMalloc(&d_seq1, n);
    cudaMalloc(&d_seq2, m);
    cudaMalloc(&d_H, size);
    // cudaMemcpy(d_seq1, seq1.data(), len1, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_seq2, seq2.data(), len2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq1, seq1, n * s, cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_seq2, seq2, m * s, cudaMemcpyHostToDevice) ;

    cudaMemcpy(d_H, h_H, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int maxDiag = n + m;
    for (int diag = 2; diag <= maxDiag; ++diag) {
        int i_start = max(1, diag - m);
        int i_end   = min(n, diag - 1);
        int count   = max(0, i_end - i_start + 1);
        if (count <= 0) continue;
        int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
        // smithWatermanDiagonal<<<blocks, threadsPerBlock>>>(
        //     reinterpret_cast<const unsigned char*>(d_seq1),
        //     reinterpret_cast<const unsigned char*>(d_seq2), 
        //     d_H, cols, diag, n, m);
        smithWatermanDiagonal<<<blocks, threadsPerBlock>>>(d_seq1, d_seq2,
            d_H, cols, diag, n, m);
            
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_H, d_H, size, cudaMemcpyDeviceToHost);

    int gpuScore = 0;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= m; ++j)
            gpuScore = std::max(gpuScore, h_H[i * cols + j]);

    // Cleanup
    free(h_H);
    cudaFree(d_seq1);
    cudaFree(d_seq2);
    cudaFree(d_H);

    return gpuScore ;

}

/*int main() {

    warmup<<<1, 1>>>() ;

    int N = 3 ;
    int length = 1000 ;
    auto avg_time = 0.0 ;

    for (int i = 0 ; i < N ; i++) {

        std::cout<<"TEST "<<i<<std::endl ;

        unsigned char *seq1 = new unsigned char[length];
        unsigned char *seq2 = new unsigned char[length] ; 
        randomSequence(length, seq1) ;
        randomSequence(length, seq2) ;

        int scoreREF = referenceSmithWaterman(seq1, seq2, length, length) ;
        std::cout<<"Reference score is "<<scoreREF<<std::endl ;

        auto start = high_resolution_clock::now() ; 
        int SCORE = SmithDiagonalGPU(seq1, seq2, length, length) ;
        auto stop = high_resolution_clock::now() ;
        auto time = duration_cast<microseconds>(stop - start) ;
        avg_time += time.count()/1000 ;
        std::cout << "Score = "<<SCORE<<", "<<time.count()/1000<<" ms " << "\n" ;
    }
    avg_time = avg_time / N ;
    std::cout << "Average time for computation is " << avg_time << "\n" ;

    return 0 ;
}*/

/*int main() {
    const int numTests = 10;
    const int seqLen = 10000;
    std::cout << "Running " << numTests << " tests on random " << seqLen << "-mer sequences..." << std::endl;
    int passed = 0;
    for (int t = 1; t <= numTests; ++t) {
        auto s1 = randomSequence(seqLen);
        auto s2 = randomSequence(seqLen);
        if (runTest(s1, s2, t)) ++passed;
    }
    std::cout << passed << "/" << numTests << " tests passed." << std::endl;
    return (passed == numTests) ? EXIT_SUCCESS : EXIT_FAILURE;
}*/