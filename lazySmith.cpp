#pragma once
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <optional>
#include <vector>
#include <iostream>
#include <algorithm>

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

void leftRotation(std::vector<int> &vector) {

    int n = vector.size() ;
    int temp = vector[0] ;
    for (int i = 0 ; i < n - 1 ; i ++) {
        vector[i] = vector[i+1] ;
    }
    vector[n-1] = temp ;
}

/**
 * returns the highest Smith-Waterman local alignment score we found between the two DNA sequences
 * initializes the matrices E, F, H of size (m+1)*(n+1)
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
            std::cout<<"debug while loop"<<std::endl ;
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


int main() {

    unsigned char seq1[] = "ABDAAADB" ;
    unsigned char seq2[] = "ADDBAABB" ;

    // unsigned char seq1[] = "ABDA" ;
    // unsigned char seq2[] = "ADDB" ;

    // unsigned char seq1[] = "AAA" ;
    // unsigned char seq2[] = "AAA" ;

    std::cout << "debug " << sizeof(seq1) << std::endl ;

    int n = sizeof(seq1) - 1 ; 
    int m = sizeof(seq2) - 1 ;

    int score = LazySmith(seq1, seq2, n, m) ;
    std::cout << "Lazy Smith Waterman Score result is " << score << std::endl ;

    return 0 ;
}