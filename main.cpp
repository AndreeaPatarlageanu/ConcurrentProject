#pragma once
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <optional>
#include <vector>
#include <iostream>

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

int main() {

    // unsigned char seq1[] = "ABDAAADB" ;
    // unsigned char seq2[] = "ADDBAABB" ;
    // unsigned char seq1[] = "ABDA" ;
    // unsigned char seq2[] = "ADDB" ;
    unsigned char seq1[] = "AAA" ;
    unsigned char seq2[] = "AAA" ;

    std::cout << "debug " << sizeof(seq1) << std::endl ;

    int n = sizeof(seq1) - 1 ; 
    int m = sizeof(seq2) - 1 ;

    int score = SmithWatermanScore(seq1, seq2, n, m) ;
    std::cout << "Smith Waterman Score result is " << score << std::endl ;

    return 0 ;
}