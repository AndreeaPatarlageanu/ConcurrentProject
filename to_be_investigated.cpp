#pragma once
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <optional>
#include <vector>
#include <iostream>
#include <algorithm>
#include <future>

#include <chrono>
using namespace std::chrono;

/**
 * constant for scoring
 * G_INIT - penalty for starting a gap
 * G_EXT - penalty for continuing a gap 
 */
const int G_INIT = 1 ;
const int G_EXT = 1 ; 
const int MATCH = 1 ;
const int MISMATCH = -1 ;

/**
 * in article, score W <= 0 if mismatch, W > 0 if match
 */
int score(unsigned char s1, unsigned char s2) {
    if (s1 == s2) {
        return MATCH;
    }
    return MISMATCH;
}

void leftRotation(std::vector<int> &vector) {

    int n = vector.size() ;
    int temp = vector[0] ;
    for (int i = 0 ; i < n - 1 ; i ++) {
        vector[i] = vector[i+1] ;
    }
    vector[n-1] = temp ;
}

/**
 * Original LazySmith function - one chunk
 */
int LazySmith(unsigned char *seq1, unsigned char *seq2, int n, int m){
    std::vector<int> vHLoad(n + 1, 0), vHStore(n + 1, 0), vE(n + 1, 0), vF(n + 1, 0);
    int segLen = (n + 15) / 16;
    int dbLen = m;
    int maxScore = 0;

    for(int i = 0; i < dbLen; i++){
        std::fill(vF.begin(), vF.end(), 0);
        // int vH = (segLen - 1 < vHStore.size()) ? vHStore[segLen - 1] << 1 : 0;
        int vH = vHStore[segLen - 1] ; // << 1;
        std::swap(vHLoad, vHStore);

        for(int j = 0; j < segLen; j++) {
            // Safely compute score
            // int s = (j < n && i < m) ? score(seq1[j], seq2[i]) : 0;
            int s = score(seq1[j], seq2[i]) ;
            vH += s;

            // Update max score
            maxScore = std::max(maxScore, vH);

            vH = std::max(vH, vE[j]);
            vH = std::max(vH, vF[j]);

            vHStore[j] = vH;

            // Calculate new vE
            // int vH_gap = vH - G_INIT;
            vH = vH - G_INIT ;
            vE[j] = vE[j] - G_EXT ;
            vE[j] = std::max(vE[j], vH) ;

            vF[j] = vF[j] - G_EXT ;
            // Calculate new vF (element-wise)
            vF[j] = std::max(vF[j], vH) ;

            // Load next H
            vH = vHLoad[j];
        }

        // // --- Lazy-F Loop ---
        leftRotation(vF) ;

        int j = 0;
        while (std::any_of(vF.begin(), vF.end(), [&](int f){ return f > (vHStore[j] - G_INIT); })) {
            //std::cout<<"debug while loop"<<std::endl ;
            vHStore[j] = std::max(vHStore[j], vF[j]);
            vF[j] -= G_EXT;

            if (++j >= segLen) {
                leftRotation(vF) ;
                j = 0;
            }
        }
    }

    return maxScore;
}


//Anti-diagonal parallelization
//Each anti-diagonal can be computed in paralle since they do not depend on others

int AntiDiagonalSmith(unsigned char *seq1, unsigned char *seq2, int n, int m, 
                     int num_threads = std::thread::hardware_concurrency()) {
    
    //base case, for small sequences we use LazySmith (default)
    if ( n <= 1 || m <= 1 || num_threads <= 1 )
            return LazySmith( seq1, seq2, n, m );
    
    // Create matrices for H, E, F scores
    std::vector<std::vector<int>> H(n + 1, std::vector<int>(m + 1, 0));
    std::vector<std::vector<int>> E(n + 1, std::vector<int>(m + 1, 0));  
    std::vector<std::vector<int>> F(n + 1, std::vector<int>(m + 1, 0));
    
    int max_score = 0;
    
    // Process each anti-diagonal
    // Total anti-diagonals = n + m - 1
    for (int diag = 1; diag <= n + m - 1; diag++) {
        
        // Find the range of cells in this anti-diagonal
        int start_i = std::max(1, diag - m);
        int end_i = std::min(diag, n);
        
        // If the diagonal is small, don't bother with threads
        int diag_size = end_i - start_i + 1;
        if (diag_size <= num_threads || diag_size <= 4) {
            // Process sequentially for small diagonals
            for (int i = start_i; i <= end_i; i++) {
                int j = diag - i;
                if (j >= 1 && j <= m) {
                    
                    // Standard Smith-Waterman recurrence
                    int match_score = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                    
                    // Update E (gap in seq2)
                    E[i][j] = std::max(E[i-1][j] - G_EXT, H[i-1][j] - G_INIT);
                    
                    // Update F (gap in seq1) 
                    F[i][j] = std::max(F[i][j-1] - G_EXT, H[i][j-1] - G_INIT);
                    
                    // Update H
                    H[i][j] = std::max({0, match_score, E[i][j], F[i][j]});
                    
                    max_score = std::max(max_score, H[i][j]);
                }
            }
        } else {
            // Process in parallel for large diagonals
            std::vector<std::future<int>> futures;
            int chunk_size = (diag_size + num_threads - 1) / num_threads;
            
            for (int t = 0; t < num_threads; t++) {
                int chunk_start = start_i + t * chunk_size;
                int chunk_end = std::min(chunk_start + chunk_size - 1, end_i);
                
                if (chunk_start <= end_i) {
                    futures.push_back(std::async(std::launch::async, 
                        [&, chunk_start, chunk_end, diag]() -> int {
                            int local_max = 0;
                            
                            for (int i = chunk_start; i <= chunk_end; i++) {
                                int j = diag - i;
                                if (j >= 1 && j <= m) {
                                    
                                    // Standard Smith-Waterman recurrence
                                    int match_score = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                                    
                                    // Update E (gap in seq2)
                                    E[i][j] = std::max(E[i-1][j] - G_EXT, H[i-1][j] - G_INIT);
                                    
                                    // Update F (gap in seq1)
                                    F[i][j] = std::max(F[i][j-1] - G_EXT, H[i][j-1] - G_INIT);
                                    
                                    // Update H
                                    H[i][j] = std::max({0, match_score, E[i][j], F[i][j]});
                                    
                                    local_max = std::max(local_max, H[i][j]);
                                }
                            }
                            return local_max;
                        }));
                }
            }
            
            // Collect results from all threads
            for (auto& future : futures) {
                max_score = std::max(max_score, future.get());
            }
        }
    }
    
    return max_score;
}

/**
 * Parallel Smith-Waterman using sequence chunking
 * @param seq1 First sequence (will be chunked)
 * @param seq2 Second sequence
 * @param n Length of seq1
 * @param m Length of seq2
 * @param num_threads Number of threads to use
 * @param chunk_strategy 0=chunk seq1, 1=chunk seq2, 2=chunk longer sequence
 */
int ParallelLazySmith( unsigned char *seq1, unsigned char *seq2, int n, int m, 
                      int num_threads = std::thread::hardware_concurrency(),
                      int chunkStrategy = 2 ) {
    
    //in case we have have only one thread, then we apply the normal version
    if( num_threads <= 1 )
        return LazySmith( seq1, seq2, n, m );
    

    //which DNA sequence to split up between threads for parallel processing
    //Strategy 0: always split sequence 1
    //Strategy 1: always split sequence 2
    //Strategy 2 (default one): split the sequence which is longer
    bool chunk_seq1 = false;

    if ( chunkStrategy == 0 )
        chunk_seq1 = true;
    else if ( chunkStrategy == 1 )
        chunk_seq1 = false;
    else {
        //here we chunk the longer sequence
        if( n >= m )
            chunk_seq1 = true;
        else chunk_seq1 = false;
    }

    std::vector<std::future<int>> futures;
    int max_global_score = 0;

    if ( chunk_seq1 == true ) {  //we work with n from now on

        int chunk_size = ( n + num_threads - 1 ) / num_threads;
        int t;

        for ( t = 0; t < num_threads; t++ ) {
            //like in the TDs
            int beginning = t * chunk_size;
            int end = std::min( beginning + chunk_size, n );
            
            if ( beginning >= n )  //base case
                break;
            
            int chunk_len = end - beginning;
            
            futures.push_back( std::async( std::launch::async, [=]() {
                return LazySmith( seq1 + beginning, seq2, chunk_len, m );
            }));
        }
    } 
    else {  //here with m

        int chunk_size = ( m + num_threads - 1 ) / num_threads;
        int t;

        for ( t = 0; t < num_threads; t++ ) {
            int beginning = t * chunk_size;
            int end = std::min( beginning + chunk_size, m );
            
            if ( beginning >= m ) 
                break;
            
            int chunk_len = end - beginning;
            
            futures.push_back( std::async( std::launch::async, [=]() {
                return LazySmith( seq1, seq2 + beginning, n, chunk_len );
            }));
        }
    }

    // Collect results from all threads
    for (auto& future : futures) {
        int chunk_score = future.get();
        max_global_score = std::max(max_global_score, chunk_score);
    }

    return max_global_score;
}

void runTests() {
    struct TestCase {
        std::string seq1, seq2;
        int expected_value;
    };

    std::vector<TestCase> tests = {
        { "ABDAAADB", "ADDBAABB", 2 },
        { "ABDA", "ADDB", 1 },
        { "AAA", "AAA", 3 },
        { "A", "A", 1 },
        { "A", "G", 0 },
        {
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG",
            1
        },
        {
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
            0
        },
        {
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
            0
        }
    };

    int passed = 0;
    std::cout << "Testing both normal and parallel versions:\n";
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads\n\n";

    for ( size_t i = 0; i < tests.size(); i++ ) {
        const auto& t = tests[ i ];
        
        //Test normal version(slow one)
        auto start_normal = high_resolution_clock::now();
        int normal_score = LazySmith(
            (unsigned char*)t.seq1.c_str(),
            (unsigned char*)t.seq2.c_str(),
            t.seq1.size(),
            t.seq2.size()
        );
        auto end_normal = high_resolution_clock::now();
        auto duration_normal = duration_cast<microseconds>( end_normal - start_normal );

        //Test parallel version(fast one)
        auto start_parallel = high_resolution_clock::now();
        int parallel_score = ParallelLazySmith(
            (unsigned char*)t.seq1.c_str(),
            (unsigned char*)t.seq2.c_str(),
            t.seq1.size(),
            t.seq2.size()
        );
        auto end_parallel = high_resolution_clock::now();
        auto duration_parallel = duration_cast<microseconds>( end_parallel - start_parallel );

        //Test new paralllel
        auto start_new = high_resolution_clock::now();
        int new_score = AntiDiagonalSmith(
            (unsigned char*)t.seq1.c_str(),
            (unsigned char*)t.seq2.c_str(),
            t.seq1.size(),
            t.seq2.size()
        );
        auto end_new = high_resolution_clock::now();
        auto duration_new = duration_cast<microseconds>( end_new - start_new );


        bool normal_correct = ( normal_score == t.expected_value );
        bool parallel_correct = ( parallel_score == t.expected_value ); // Changed: exact match required
        bool new_correct = ( new_score == t.expected_value ); // Changed: exact match required

        if ( normal_correct )
            passed++;

        std::cout << "Test " << i + 1 << ":\n";

        std::cout << "  Serial:   "
                  << " Score=" << normal_score << " (expected=" << t.expected_value << ")"
                  << " Time=" << duration_normal.count() / 1000.0 << "ms";

        std::cout << "  Parallel: "
                  << " Score=" << parallel_score 
                  << " Time=" << duration_parallel.count() / 1000.0 << "ms"
                  << " Speedup=" << (double)duration_normal.count() / duration_parallel.count() << "x";

        std::cout << "  New Parallel: "
                  << " Score=" << new_score 
                  << " Time=" << duration_new.count() / 1000.0 << "ms"
                  << " Speedup=" << (double)duration_normal.count() / duration_new.count() << "x";
    }

    std::cout << "Summary: " << passed << "/" << tests.size() << " serial tests passed.\n";
}

int main() {

    runTests();
    return 0 ;
}
