#pragma once
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <optional>
#include <vector>
#include <iostream>
//#include "tests.h"

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
int score(unsigned char s1, unsigned char s2) {
    if (s1 == s2) {
        return MATCH ;
    }
    return MISMATCH ;
}

/**
 * constructs the DP matrices 
 * using algo in https://academic.oup.com/bioinformatics/article/23/2/156/205631
 * @arg the pointer to the three matrices we want to construct (passed by reference), q and d are the two sequences to compare with respective lengths n and m
 */
void dpMatrices(std::vector<std::vector<int>> &E, std::vector<std::vector<int>> &F, std::vector<std::vector<int>> &H, 
    unsigned char *q, unsigned char *d, int n, int m){

    for (int j = 0 ; j < m + 1 ; j++) {
        E[j][0] = 0 ;
        F[j][0] = 0 ;
        H[j][0] = 0 ;
    }
    for (int j = 0 ; j < n + 1 ; j++) {
        E[0][j] = 0 ;
        F[0][j] = 0 ;
        H[0][j] = 0 ;
    }

    for (int i = 1 ; i < m + 1 ; i ++) {
        for (int j = 1 ; j < n + 1 ; j++) {

            E[i][j] = std::max( E[i][j-1] - G_EXT, H[i][j-1] - G_INIT) ;
            F[i][j] = std::max( F[i-1][j] - G_EXT, H[i-1][j] - G_INIT) ;

            // my code
            int temp1 = std::max ( F[i][j], E[i][j] ) ;
            int temp2 = std::max( 0, H[i-1][j-1] + score(q[j-1], d[i-1])) ; // careful to index! i and j start at 1
            H[i][j] = std::max( temp1, temp2 ) ;

        }
    }

}

/**
 * returns the highest Smith-Waterman local alignment score we found between the two DNA sequences
 * initializes the matrices E, F, H of size (m+1)*(n+1)
 */
int SmithWatermanScore(unsigned char *seq1, unsigned char *seq2, int n, int m){

    std::vector<std::vector<int>> E(m + 1, std::vector<int>(n + 1)) ;
    std::vector<std::vector<int>> F(m + 1, std::vector<int>(n + 1)) ;
    std::vector<std::vector<int>> H(m + 1, std::vector<int>(n + 1)) ;

    dpMatrices(E, F, H, seq1, seq2, n, m) ;

    int maxScore = 0;
    for (int i = 0 ; i < m+1 ; ++i) {
        for (int j = 0 ; j < n+1 ; ++j) {
            maxScore = std::max(maxScore, H[i][j]) ;
        }
    }

    return maxScore ;
}

// void runTests() {
//     struct TestCase {
//         std::string seq1, seq2;
//         int expected_value;
//     };

//     std::vector<TestCase> tests = {
//         { "ABDAAADB", "ADDBAABB", 2 },
//         { "ABDA", "ADDB", 1 },
//         { "AAA", "AAA", 3 },
//         { "A", "A", 1 },
//         { "A", "G", 0 },
//         {
//             "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
//             "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG",
//             1
//         }
//     };

//     int passed = 0;
//     std::cout << "Starting " << tests.size() << " tests:\n";

//     for (size_t i = 0; i < tests.size(); ++i) {
//         const auto& t = tests[i];
//         auto start = high_resolution_clock::now();

//         int score = SmithWatermanScore(
//             (unsigned char*)t.seq1.c_str(),
//             (unsigned char*)t.seq2.c_str(),
//             t.seq1.size(),
//             t.seq2.size()
//         );

//         auto end = high_resolution_clock::now();
//         auto duration = duration_cast<microseconds>(end - start);

//         bool correct = (score == t.expected_value);
//         if (correct) ++passed;

//         std::cout << "Test " << i + 1 << ": " << (correct ? "Passed" : "Failed")
//                   << " | Expected = " << t.expected_value
//                   << ", Got = " << score
//                   << " | Time = " << duration.count() / 1000.0 << " ms\n";
//     }

//     std::cout << "\nSummary: " << passed << "/" << tests.size() << " tests passed.\n";
// }

// int main() {

//     // auto start = high_resolution_clock::now() ; 

//     // // unsigned char seq1[] = "ABDAAADB" ;
//     // // unsigned char seq2[] = "ADDBAABB" ;

//     // // unsigned char seq1[] = "ABDA" ;
//     // // unsigned char seq2[] = "ADDB" ;

//     // // unsigned char seq1[] = "AAA" ;
//     // // unsigned char seq2[] = "AAA" ;

//     // unsigned char seq1[] = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" ;
//     // unsigned char seq2[] = "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG" ;

//     // std::cout << "debug " << sizeof(seq1) << std::endl ;

//     // int n = sizeof(seq1) - 1 ; 
//     // int m = sizeof(seq2) - 1 ;

//     // int score = SmithWatermanScore(seq1, seq2, n, m) ;
//     // std::cout << "Smith Waterman Score result is " << score << std::endl ;

//     // auto stop = high_resolution_clock::now() ;
// 	// auto time = duration_cast<microseconds>(stop - start) ;
// 	// std::cout << "Time for computation is " << time.count()/1000 << "\n" ;

//     // return 0 ;
//     runTests();
//     return 0;
// }

// g++ -std=c++17 -O2 -o smith_test main.cpp
// ./smith_test
