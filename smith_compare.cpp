#include <climits>
#include <thread>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

/**
 * Scoring constants
 */
const int G_INIT = 1;
const int G_EXT = 1;
const int MATCH = 1;
const int MISMATCH = -1;

/**
 * Match score function
 */
int score(unsigned char s1, unsigned char s2) {
    return (s1 == s2) ? MATCH : MISMATCH;
}

/**
 * Sequential Smith-Waterman DP matrix fill
 */
void dpSequential(std::vector<std::vector<int>> &E,
                  std::vector<std::vector<int>> &F,
                  std::vector<std::vector<int>> &H,
                  unsigned char *q, unsigned char *d,
                  int n, int m) {

    for (int j = 0; j < m + 1; j++) E[j][0] = F[j][0] = H[j][0] = 0;
    for (int j = 0; j < n + 1; j++) E[0][j] = F[0][j] = H[0][j] = 0;

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
 * Threaded wavefront parallelization of Smith-Waterman
 */
void dpThreaded(std::vector<std::vector<int>> &E,
                std::vector<std::vector<int>> &F,
                std::vector<std::vector<int>> &H,
                unsigned char *q, unsigned char *d,
                int n, int m) {

    for (int j = 0; j < m + 1; j++) E[j][0] = F[j][0] = H[j][0] = 0;
    for (int j = 0; j < n + 1; j++) E[0][j] = F[0][j] = H[0][j] = 0;

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

        for (auto &t : threads) t.join();
    }
}

/**
 * Wrapper for both versions
 */
int SmithWatermanScore(unsigned char *seq1, unsigned char *seq2, int n, int m, bool threaded) {
    std::vector<std::vector<int>> E(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<int>> F(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<int>> H(m + 1, std::vector<int>(n + 1));

    if (threaded) {
        dpThreaded(E, F, H, seq1, seq2, n, m);
    } else {
        dpSequential(E, F, H, seq1, seq2, n, m);
    }

    int maxScore = 0;
    for (int i = 0; i < m + 1; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            maxScore = std::max(maxScore, H[i][j]);
        }
    }

    return maxScore;
}

/**
 * Main function for comparison
 */
int main() {
    // Change these test cases
    unsigned char seq1[] = "TACGGGCCCGCTAC";
    unsigned char seq2[] = "TAGCCCTATCGGTCA";

    int n = sizeof(seq1) - 1;
    int m = sizeof(seq2) - 1;

    std::cout << "Comparing sequential vs threaded Smith-Waterman:\n";
    std::cout << "seq1: " << seq1 << "\nseq2: " << seq2 << "\n";

    // Sequential
    auto start_seq = std::chrono::high_resolution_clock::now();
    int score_seq = SmithWatermanScore(seq1, seq2, n, m, false);
    auto end_seq = std::chrono::high_resolution_clock::now();

    // Threaded
    auto start_thr = std::chrono::high_resolution_clock::now();
    int score_thr = SmithWatermanScore(seq1, seq2, n, m, true);
    auto end_thr = std::chrono::high_resolution_clock::now();

    auto dur_seq = std::chrono::duration_cast<std::chrono::microseconds>(end_seq - start_seq).count();
    auto dur_thr = std::chrono::duration_cast<std::chrono::microseconds>(end_thr - start_thr).count();

    std::cout << "\nSequential score: " << score_seq << " (" << dur_seq << " µs)\n";
    std::cout << "Threaded score:   " << score_thr << " (" << dur_thr << " µs)\n";

    return 0;
}