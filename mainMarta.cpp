#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <optional>
#include <vector>
#include <iostream>

/**
 * Constants for scoring:
 * G_INIT - penalty for starting a gap
 * G_EXT  - penalty for continuing a gap
 * MATCH  - score for match
 * MISMATCH - score for mismatch
 */
const int G_INIT = 1;
const int G_EXT = 1;
const int MATCH = 1;
const int MISMATCH = -1;

/*
Recommended online literature:
*/
// const int MATCH = 2;
// const int MISMATCH = -1;
// const int G_INIT = 2;
// const int G_EXT = 1;

/**
 * Returns score for two characters:
 * +1 if match, -1 if mismatch
 */
int score(unsigned char s1, unsigned char s2) {
    return (s1 == s2) ? MATCH : MISMATCH;
}

/**
 * Constructs the dynamic programming matrices:
 * E, F - for gap penalties
 * H - for alignment score
 */
void dpMatrices(std::vector<std::vector<int> > &E,
                std::vector<std::vector<int> > &F,
                std::vector<std::vector<int> > &H,
                unsigned char *q, unsigned char *d,
                int n, int m) {

    for (int j = 0; j < m + 1; j++) {
        E[j][0] = 0;
        F[j][0] = 0;
        H[j][0] = 0;
    }
    for (int j = 0; j < n + 1; j++) {
        E[0][j] = 0;
        F[0][j] = 0;
        H[0][j] = 0;
    }

    for (int i = 1; i < m + 1; i++) {
        for (int j = 1; j < n + 1; j++) {
            E[i][j] = std::max(E[i][j - 1] - G_EXT, H[i][j - 1] - G_INIT);
            F[i][j] = std::max(F[i - 1][j] - G_EXT, H[i - 1][j] - G_INIT);

            int temp1 = std::max(F[i][j], E[i][j]);
            int temp2 = std::max(0, H[i - 1][j - 1] + score(q[j - 1], d[i - 1]));
            H[i][j] = std::max(temp1, temp2);
        }
    }
}

/**
 * Computes the Smith-Waterman score using DP matrices
 */
int SmithWatermanScore(unsigned char *seq1, unsigned char *seq2, int n, int m) {
    std::vector<std::vector<int> > E(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<int> > F(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<int> > H(m + 1, std::vector<int>(n + 1));

    dpMatrices(E, F, H, seq1, seq2, n, m);

    int maxScore = 0;
    for (int i = 0; i < m + 1; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            maxScore = std::max(maxScore, H[i][j]);
        }
    }

    return maxScore;
}

int main() {
    // Test sequences — try different ones here
    // Test Case 1
// unsigned char seq1[] = "AAA";
// unsigned char seq2[] = "AAA"; // Expected score: 3

// Test Case 2
// unsigned char seq1[] = "AAA";
// unsigned char seq2[] = "AAB"; // Expected score: 2

// Test Case 3
// unsigned char seq1[] = "GATTACA";
// unsigned char seq2[] = "GCATGCU"; // Expected score: 2

// Test Case 4
// unsigned char seq1[] = "ACACACTA";
// unsigned char seq2[] = "AGCACACA"; // Expected score: 5

// Test Case 5
// unsigned char seq1[] = "AGGGCT";
// unsigned char seq2[] = "AGGCA"; // Expected score: 4

// Test Case 6
// unsigned char seq1[] = "TTAC";
// unsigned char seq2[] = "GTTACG"; // Expected score: 4

// Test Case 7
// unsigned char seq1[] = "ACTGATTCA";
// unsigned char seq2[] = "ACCGTGCGA"; // Expected score: 4

// Test Case 8
// unsigned char seq1[] = "ACGT";
// unsigned char seq2[] = "TGCA"; // Expected score: 1

// Test Case 9
unsigned char seq1[] = "TACGGGCCCGCTAC";
unsigned char seq2[] = "TAGCCCTATCGGTCA"; // Expected score: 7

// Test Case 10
// unsigned char seq1[] = "AGTACGCA";
// unsigned char seq2[] = "TATGC"; // Expected score: 3

    std::cout << "debug " << sizeof(seq1) << std::endl;

    int n = sizeof(seq1) - 1;
    int m = sizeof(seq2) - 1;

    std::cout << "seq1: " << seq1 << "; seq2: " << seq2 << std::endl;

    int score = SmithWatermanScore(seq1, seq2, n, m);
    std::cout << "Smith Waterman Score result is " << score << std::endl;

    return 0;
}