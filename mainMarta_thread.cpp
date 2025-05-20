#include <climits>
#include <thread>
#include <vector>
#include <iostream>
#include <algorithm>

/**
 * Constants for scoring:
 */
const int G_INIT = 1;
const int G_EXT = 1;
const int MATCH = 1;
const int MISMATCH = -1;

/**
 * Score function for local alignment
 */
int score(unsigned char s1, unsigned char s2) {
    return (s1 == s2) ? MATCH : MISMATCH;
}

/**
 * Parallelized DP matrix construction using std::thread
 */
void dpMatrices(std::vector<std::vector<int>> &E,
                std::vector<std::vector<int>> &F,
                std::vector<std::vector<int>> &H,
                unsigned char *q, unsigned char *d,
                int n, int m) {

    for (int j = 0; j < m + 1; j++) {
        E[j][0] = F[j][0] = H[j][0] = 0;
    }
    for (int j = 0; j < n + 1; j++) {
        E[0][j] = F[0][j] = H[0][j] = 0;
    }

    for (int k = 2; k <= m + n; ++k) {
        std::vector<std::thread> threads;

        for (int i = std::max(1, k - n); i <= std::min(m, k - 1); ++i) {
            int j = k - i;

            threads.emplace_back([&, i, j]() {
                E[i][j] = std::max(E[i][j - 1] - G_EXT, H[i][j - 1] - G_INIT);
                F[i][j] = std::max(F[i - 1][j] - G_EXT, H[i - 1][j] - G_INIT);

                int temp1 = std::max(F[i][j], E[i][j]);
                int temp2 = std::max(0, H[i - 1][j - 1] + score(q[j - 1], d[i - 1]));
                H[i][j] = std::max(temp1, temp2);
            });
        }

        for (auto &t : threads) {
            t.join(); // Wait for all threads in this diagonal
        }
    }
}

/**
 * Compute Smith-Waterman score between two sequences
 */
int SmithWatermanScore(unsigned char *seq1, unsigned char *seq2, int n, int m) {
    std::vector<std::vector<int>> E(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<int>> F(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<int>> H(m + 1, std::vector<int>(n + 1));

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
    // Example test case
    unsigned char seq1[] = "TACGGGCCCGCTAC";
    unsigned char seq2[] = "TAGCCCTATCGGTCA"; // Expected score: 4 under current scoring

    std::cout << "debug " << sizeof(seq1) << std::endl;

    int n = sizeof(seq1) - 1;
    int m = sizeof(seq2) - 1;

    std::cout << "seq1: " << seq1 << "; seq2: " << seq2 << std::endl;

    int score = SmithWatermanScore(seq1, seq2, n, m);
    std::cout << "Smith Waterman Score result is " << score << std::endl;

    return 0;
}