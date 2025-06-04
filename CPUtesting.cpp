// Test scores for CPU implementations

#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <optional>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>

using namespace std;
using namespace std::chrono;

const int G_INIT = 1;
const int G_EXT = 1;
const int MATCH = 1;
const int MISMATCH = -1;

int score(unsigned char s1, unsigned char s2) {
    return (s1 == s2) ? MATCH : MISMATCH;
}

int LazySmith(unsigned char *seq1, unsigned char *seq2, int n, int m) {
    std::vector<int> H_prev(n+1, 0), H_curr(n+1, 0);
    std::vector<int> E(n+1, 0);
    std::vector<int> F(n+1, 0);

    int maxScore = 0;

    for (int i = 1; i <= m; ++i) {
        E[0]      = 0;
        F[0]      = 0;
        H_curr[0] = 0;

        for (int j = 1; j <= n; ++j) {
            E[j] = std::max(E[j-1] - G_EXT, H_curr[j-1] - G_INIT);

            int Ht = H_prev[j-1] + score(seq1[j-1], seq2[i-1]);

            F[j] = std::max(F[j] - G_EXT, H_prev[j] - G_INIT);

            int hval = Ht;
            if (E[j] > hval)  hval = E[j];
            if (F[j] > hval)  hval = F[j];
            if (hval < 0)     hval = 0;
            H_curr[j] = hval;

            if (hval > maxScore) maxScore = hval;
        }

        for (int j = 1; j <= n; ++j) {
            int fij = H_curr[j] - G_INIT;
            if (fij > F[j]) {
                F[j] = fij;
                if (F[j] > H_curr[j]) {
                    H_curr[j] = F[j];
                    if (H_curr[j] > maxScore) maxScore = H_curr[j];
                }
                for (int k = j+1; k <= n; ++k) {
                    int newF = F[k-1] - G_EXT;
                    if (newF <= 0) break;
                    if (newF <= F[k]) break;
                    F[k] = newF;
                    if (F[k] > H_curr[k]) {
                        H_curr[k] = F[k];
                        if (H_curr[k] > maxScore) maxScore = H_curr[k];
                    }
                }
            }
        }

        std::swap(H_prev, H_curr);
        std::fill(H_curr.begin(), H_curr.end(), 0);
    }

    return maxScore;
}

void dpMatrices(std::vector<std::vector<int>> &E,
                std::vector<std::vector<int>> &F,
                std::vector<std::vector<int>> &H,
                unsigned char *q, unsigned char *d,
                int n, int m) {
    for (int j = 0; j < m + 1; j++) {
        E[j][0] = 0; F[j][0] = 0; H[j][0] = 0;
    }
    for (int j = 0; j < n + 1; j++) {
        E[0][j] = 0; F[0][j] = 0; H[0][j] = 0;
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

int SmithWatermanScore(unsigned char *seq1, unsigned char *seq2, int n, int m) {
    std::vector<std::vector<int>> E(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<int>> F(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<int>> H(m + 1, std::vector<int>(n + 1));

    dpMatrices(E, F, H, seq1, seq2, n, m);

    int maxScore = 0;
    for (int i = 0; i < m + 1; ++i)
        for (int j = 0; j < n + 1; ++j)
            maxScore = std::max(maxScore, H[i][j]);

    return maxScore;
}

string randomDNA(int len, mt19937_64 &rng) {
    static const char *nts = "ACGT";
    uniform_int_distribution<int> dist(0, 3);
    string s;
    for (int i = 0; i < len; ++i) s += nts[dist(rng)];
    return s;
}

int main() {
    mt19937_64 rng(123); // fixed seed for reproducibility
    vector<pair<string, string>> tests;

    // --- Add 3 printed test cases of length 1000
    for (int i = 0; i < 3; ++i) {
        string s1 = randomDNA(1000, rng);
        string s2 = randomDNA(1000, rng);
        cout << "Test " << i+1 << ":\n";
        cout << "Seq1: " << s1 << "\n";
        cout << "Seq2: " << s2 << "\n\n";
        tests.emplace_back(s1, s2);
    }

    // --- Run tests
    for (size_t i = 0; i < tests.size(); ++i) {
        const auto &[s1, s2] = tests[i];
        vector<unsigned char> seq1(s1.begin(), s1.end()), seq2(s2.begin(), s2.end());
        int n = seq1.size(), m = seq2.size();

        auto t1 = high_resolution_clock::now();
        int score_lazy = LazySmith(seq1.data(), seq2.data(), n, m);
        auto t2 = high_resolution_clock::now();
        int score_dp = SmithWatermanScore(seq1.data(), seq2.data(), n, m);
        auto t3 = high_resolution_clock::now();

        auto lazy_t = duration_cast<milliseconds>(t2 - t1).count();
        auto dp_t = duration_cast<milliseconds>(t3 - t2).count();

        cout << "Scoring Test " << i + 1 << " (" << n << " x " << m << "):\n";
        cout << "  Lazy CPU Score = " << score_lazy << " | Time = " << lazy_t << " ms\n";
        cout << "  DP   CPU Score = " << score_dp   << " | Time = " << dp_t   << " ms\n";
        cout << endl;
    }

    return 0;
}
