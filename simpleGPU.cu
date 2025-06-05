#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <optional>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#include <chrono>
using namespace std::chrono ;

/**
 * constant for scoring
 * G_INIT - penalty for starting a gap
 * G_EXT - penalty for continuing a gap 
 * HERE CHOOSE CAREFULLY CONSTANTS 
 */
const int G_INIT = 1 ;
const int G_EXT = 1 ; 
const int MATCH = 1 ;
const int MISMATCH = -1 ;

/**
 * in article, score W <= 0 if mismatch, W > 0 if match
 */
__device__
int score(unsigned char s1, unsigned char s2) {
    if (s1 == s2) {
        return MATCH ;
    }
    return MISMATCH ;
}

__device__
int maxElement(int e1, int e2) {
    if (e1 >= e2) {
        return e1 ;
    }
    return e2 ;
}

__global__
void warmup() {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ 
void initializeM(int* E, int* F, int* H, int n, int m){

    int index = blockIdx.x * blockDim.x + threadIdx.x ;

    // first element of each row ~ first column is initialized to 0
    if (index < m + 1) {
        E[index * (n + 1) + 0] = 0 ;
        F[index * (n + 1) + 0] = 0 ;
        H[index * (n + 1) + 0] = 0 ;
    }

    // first element of each colum ~ first row is initialized to 0
    if (index < n + 1) {
        E[index] = 0 ;
        F[index] = 0 ;
        H[index] = 0 ;
    }

}

/**
 * we must be careful because the computation of each cell depends on
 * the left neighbour (for E and H), the top neigbour (for F and H) and
 * the top-left neighbour (for H) that is the previous element in the diagonal. 
 */

__global__ 
// void DPMatrices(unsigned char *seq1, unsigned char *seq2, int *E, int *F, int *H, int n, int m, int diag) {
void DPMatrices(unsigned char *seq1, unsigned char *seq2, int *E, int *F, int *H, int n, int m, int diag) {

    int index = threadIdx.x + blockIdx.x * blockDim.x ;
    int row = index + 1 ;
    int column = diag - row ;
    int id = row * (n+1) + column ;

    int r = 0 ;

    if (row >= 1 && row <= m && column >= 1 && column <= n) {

        int index_prevRow = (row - 1) * (n+1) + column ;
        int index_prevCol = row * (n+1) + (column - 1) ;
        int index_prevColRow = (row - 1) * (n+1) + (column - 1) ;

        int e = maxElement(E[index_prevCol] - G_EXT, H[index_prevCol] - G_INIT) ;
        int f = maxElement(F[index_prevRow] - G_EXT, H[index_prevRow] - G_INIT) ;

        int s = score(seq1[column - 1], seq2[row - 1]) ;
        int temp1 = maxElement(e, f) ;
        int temp2 = maxElement(0, H[index_prevColRow] + s) ;
        int h = maxElement( temp1, temp2 ) ;

        E[id] = e ;
        F[id] = f ;
        H[id] = h ;

    }

}

extern "C" int SequentialSmithWatermanScoreGPU(unsigned char *seq1, unsigned char *seq2, int n, int m) {

    int blockSize = 256 ;
    int maxSize = max(n+1, m+1) ;
    int numBlocks = (maxSize + blockSize - 1) / blockSize ;

    // contain the three matrices of size (m+1) * (n+1)
    int *vE ;
    int *vF ;
    int *vH ;
    cudaMalloc((void **)&vE, (m+1) * (n+1) * sizeof(int)) ;
    cudaMalloc((void **)&vF, (m+1) * (n+1) * sizeof(int)) ;
    cudaMalloc((void **)&vH, (m+1) * (n+1) * sizeof(int)) ;

    size_t s = sizeof(unsigned char) ;

    unsigned char *seq1_copy ;
    unsigned char *seq2_copy ;

    cudaMalloc((void **)&seq1_copy, n * s) ;
    cudaMalloc((void **)&seq2_copy, m * s) ;
    cudaMemcpy(seq1_copy, seq1, n * s, cudaMemcpyHostToDevice) ;
    cudaMemcpy(seq2_copy, seq2, m * s, cudaMemcpyHostToDevice) ;

    // Initialize matrices
    initializeM<<<numBlocks, blockSize>>>(vE, vF, vH, n, m) ;
    cudaDeviceSynchronize() ;

    // WAVEFRONT APPROACH
    for (int diag = 2; diag <= m + n; ++diag) {
        int numCells = min(diag - 1, min(m, n)) ; // max number of elements on this diagonal
        int numBlocks = (numCells + blockSize - 1) / blockSize ;

        DPMatrices<<<numBlocks, blockSize>>>(seq1_copy, seq2_copy, vE, vF, vH, n, m, diag) ;
        cudaDeviceSynchronize() ; // we need to make sure the previous anti-diagonal is entirely computed
    }

    int *Hresult = new int[(m+1) * (n+1)] ;
    cudaMemcpy(Hresult, vH, (m+1) * (n+1) * sizeof(int), cudaMemcpyDeviceToHost) ;

    int S = 0 ;
    for (int index = 0 ; index < (m+1) * (n+1) ; index ++) {
        S = std::max(S, Hresult[index]) ;
    }

    cudaFree(seq1_copy) ;
    cudaFree(seq2_copy) ;
    cudaFree(vE) ;
    cudaFree(vF) ;
    cudaFree(vH) ;
    delete[] Hresult ;

    return S ;

}

/*int main() {

    // to warm up the GPU as we did in TD3
    warmup<<<1, 1>>>() ;

    unsigned char seq1_1[] = "ABDAAADB" ;
    unsigned char seq2_1[] = "ADDBAABB" ;

    unsigned char seq1_2[] = "ABDA" ;
    unsigned char seq2_2[] = "ADDB" ;

    unsigned char seq1_3[] = "AAA" ;
    unsigned char seq2_3[] = "AAA" ;

    unsigned char seq1_4[] = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" ;
    unsigned char seq2_4[] = "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG" ;

    unsigned char seq1_5[] = "AAA" ;
    unsigned char seq2_5[] = "BBB" ;

    // std::cout << "debug " << sizeof(seq1) << std::endl ;

    int n_1 = sizeof(seq1_1) - 1 ; 
    int m_1 = sizeof(seq2_1) - 1 ;

    int n_2 = sizeof(seq1_2) - 1 ; 
    int m_2 = sizeof(seq2_2) - 1 ;

    int n_3 = sizeof(seq1_3) - 1 ; 
    int m_3 = sizeof(seq2_3) - 1 ;

    int n_4 = sizeof(seq1_4) - 1 ; 
    int m_4 = sizeof(seq2_4) - 1 ;

    int n_5 = sizeof(seq1_5) - 1 ; 
    int m_5 = sizeof(seq2_5) - 1 ;

    auto start_1 = high_resolution_clock::now() ; 
    std::cout << "TEST 1" << std::endl ;
    int score_1 = SequentialSmithWatermanScoreGPU(seq1_1, seq2_1, n_1, m_1) ;
    std::cout << "Smith Waterman Score result is " << score_1 << std::endl ;
    auto stop_1 = high_resolution_clock::now() ;
	auto time_1 = duration_cast<microseconds>(stop_1 - start_1) ;
	std::cout << "Time for computation is " << time_1.count()/1000 << "\n" ;

    auto start_2 = high_resolution_clock::now() ; 
    std::cout << "TEST 2" << std::endl ;
    int score_2 = SequentialSmithWatermanScoreGPU(seq1_2, seq2_2, n_2, m_2) ;
    std::cout << "Smith Waterman Score result is " << score_2 << std::endl ;
    auto stop_2 = high_resolution_clock::now() ;
	auto time_2 = duration_cast<microseconds>(stop_2 - start_2) ;
	std::cout << "Time for computation is " << time_2.count()/1000 << "\n" ;

    auto start_3 = high_resolution_clock::now() ; 
    std::cout << "TEST 3" << std::endl ;
    int score_3 = SequentialSmithWatermanScoreGPU(seq1_3, seq2_3, n_3, m_3) ;
    std::cout << "Smith Waterman Score result is " << score_3 << std::endl ;
    auto stop_3 = high_resolution_clock::now() ;
	auto time_3 = duration_cast<microseconds>(stop_3 - start_3) ;
	std::cout << "Time for computation is " << time_3.count()/1000 << "\n" ;

    auto start_4 = high_resolution_clock::now() ; 
    std::cout << "TEST 4" << std::endl ;
    int score_4 = SequentialSmithWatermanScoreGPU(seq1_4, seq2_4, n_4, m_4) ;
    std::cout << "Smith Waterman Score result is " << score_4 << std::endl ;
    auto stop_4 = high_resolution_clock::now() ;
	auto time_4 = duration_cast<microseconds>(stop_4 - start_4) ;
	std::cout << "Time for computation is " << time_4.count()/1000 << "\n" ;

    auto start_5 = high_resolution_clock::now() ; 
    std::cout << "TEST 5" << std::endl ;
    int score_5 = SequentialSmithWatermanScoreGPU(seq1_5, seq2_5, n_5, m_5) ;
    std::cout << "Smith Waterman Score result is " << score_5 << std::endl ;
    auto stop_5 = high_resolution_clock::now() ;
	auto time_5 = duration_cast<microseconds>(stop_5 - start_5) ;
	std::cout << "Time for computation is " << time_5.count()/1000 << "\n" ;

    return 0 ;
} ;*/
