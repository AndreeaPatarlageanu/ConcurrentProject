#include <climits>
#include <vector>
#include <iostream>
#include <algorithm>

const int G_INIT  = 1;
const int G_EXT   = 1;
const int MATCH   = 1;
const int MISMATCH = -1;

int score2(unsigned char s1, unsigned char s2) {
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

            int Ht = H_prev[j-1] + score2(seq1[j-1], seq2[i-1]);

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

// void runTests() {
//     struct TestCase {
//         std::string seq1, seq2;
//         int expected_value;
//     };

//     std::vector<TestCase> tests = {
//         { "ABDAAADB", "ADDBAABB", 2 },
//         { "ABDA",     "ADDB",     1 },
//         { "AAA",      "AAA",      3 },
//         { "A",        "A",        1 },
//         { "A",        "G",        0 },
//         {
//             "CGTGAATCTCCGAGGTTGCTATAAGTGCATGTGTCGGAACAAGACAGACTCTTAGGTGCTGTGACCATCGCGAATGCCCCCCTGGTCAACGACGTCTATGTGTTTTACAGCTTCAGCTTCAAAAGTTTGCCATCTGTTGTTCCCTACCGTTAGGTGTGTAGATCTGGCTCCCTTCCGCATGCACATAGTTTGGGTCGGGTCCATCCTAGTCACGATATCCATGTGTCCCAAGGAGTTTAATTTCACTGGTGGCGCTGGAAGACTAAAATTGACTAGCTTGGCGTAAATCCTCGATACCACCGAACGCTCACGGTATAGCGAGTACACGATCATTCAGGAAGGAGTACCGTCTCCAACACCACAGTACGACCTGTGAAACAGAAGCTCCGGAACCGGAAATTATCCGAAGAGTCTGATGCTGTCCTGGTGGGGTGTAACATCGACGCGCCTTTGCTTAAGCCTGCTCGCGCTCTGTGACAGACAAGCGCTGCGTGCTGGAACATCACCATTTAGGTTCGGGGCCTCCGAGTCGTGCTCATGCGCGCAAGTTGAGCGATTTCAATACAAGCGATACTAAACGGAAACTGTATACATCTGTTCCCGTGGCTGGTCGTTTTGGACTGATTTACCTTTCGATTAACCGTGCTTGCTTTTAAACGGCAGGCCACCCTGCGTTGCACTAGACGGAAAAAGCGTAGCGCGTCGCCCCAGATTAGGCGGTAGTGCAGGAGTTGTTTCAGTTTGTTCGACACGGGCACCGTCGCGGGCGGGAGCATTCGGATTAAACGGGCCCTATCGCTTAGATGGTGGGTGGATAAGAATGTAGACCTCCTATAATTCGAGGGATGGAGAAACGAGGTACCGGGCCTGAGAGCCGACTAATGGGGAGCGTACCGTGATGTGTGGTTACCGTCGTTCGCACCATACATTAGGTTATACCATCTTGAGACGTATCGCGCGGGCACAAAATTAATGACGTGCATAAATTGCATGGGACATA",
//             "ATGGTAGTAGTGGTACAAAAAGTTGTTAAGTGTAATGACCCGGCCAAGGGCTCTGTTCTATACGCTCAGCTGCCCTTGGCGGTCAAACATATCTAGATCTGGCTGGAATACGATCCTGCGTAGACAACTTTTCATCCACTATATCGACGCATCGTGTGATCTCAAAACGGGCAGTTGTATGATTATAGTGATCTGTCTTAGACTAGCTATCCGTATTCAGAAGCTCGTTCTTCGGATATGAGAGGGGTGACAGCAACACTGAGTGGGTAGCCAGCGGTTATAAAAGCCGGCGGTGGCTTCGTGAAATTAAACACATCGCTGTAGAATTCCTTTGAACAAAATTGAGCAGTGTTAGGTCAAGTAGTCATCATTACTCGAGCCCGACGCCTTTCATCCCCCGGGGTTGTATAACGTCTCCGGACTACGCCGGGAGGACTCTCCGAAGTAATTTGTTATAACCAAATTCGCAAAAGTCGTACTCATTATTGCTGTTTTGCCATTTTCACCATACTCAACAACGACTCAAAACTCCCGGGACCCAGGAGGCTCACTGAATGAGCGTCATTCACATGAATGTGAAGCCCTCAGCATAACGGTGTGACAGCGTAAGGTTTAACACCTATGGACACCCGTGCGACTACCTTCAATGCGCGCAGGCATTCACACGGTCCAACCCAGGCTGGCTTCCGTGATCACTCGACTGCCGGTTTACACCGAATGTACTAGCACGAACCCGGATACCGTGACTAGTAATAGATTCCGTCCGTCGTTGTGTAACTCACGGTGCCTCATCCATGAGGGAAGGCAATCATTCCAGCTGGCCTCAGCCCTATATGAAGTAAGTAGCACGGAAGAGTACCAACGGAAACAGCTGCTCTGAAACGCGGCGATAAAAATTTCTTGTTAGATTGTACAATGCAGATCGCGAGCGTGAGAAATACTCACAGATGTCTAGCCGTGCCCTGGAACACAAGAGACACGTACTACTGTCATACAGAGC",
//             126
//         }
//     };

//     int passed = 0;
//     std::cout << "Starting " << tests.size() << " tests:\n";

//     for (size_t i = 0; i < tests.size(); ++i) {
//         const auto& t = tests[i];
//         int score = LazySmith(
//             (unsigned char*)t.seq1.c_str(),
//             (unsigned char*)t.seq2.c_str(),
//             (int)t.seq1.size(),
//             (int)t.seq2.size()
//         );
//         bool correct = (score == t.expected_value);
//         if (correct) ++passed;

//         std::cout << "Test " << i + 1 << ": "
//                   << (correct ? "Passed" : "Failed")
//                   << "  Expected=" << t.expected_value
//                   << ", Got=" << score
//                   << "\n";
//     }

//     std::cout << "\nSummary: " << passed << "/" << tests.size() << " tests passed.\n";
// }

// int main() {
//     runTests();
//     return 0;
// }
