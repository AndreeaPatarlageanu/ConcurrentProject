#include <vector>
#include <random> 
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "allAlgo.h"

#define TESTING 1
#define MATCH 1 ;
#define MISMATCH -1 ;
#define GAP_INIT 1 ;
#define GAP_EXT 1 ;

#define nucleotides "ACGT"

static std :: default_random_engine engine(8) ; 
int U(int N) {
    std::uniform_int_distribution<int> dist(0, N - 1);
    return dist(engine);
}

unsigned char randomCharacter() {

    // std::srand(std::time(0)) ;
    static bool seeded = false;
    if (!seeded) {
        std::srand(std::time(0));
        seeded = true;
    }
    char randomChar = nucleotides[std::rand() % (sizeof(nucleotides) - 1)] ;
    return randomChar ;

}

void randomSequence(int N, unsigned char* seq ) {

    for( int i = 0; i < N; i++ ) {
        seq[i] = randomCharacter() ;
    }

}

void generateTest(int N, int numb_test) {

    std::vector<std::pair<std::vector<unsigned char>, std::vector<unsigned char> > > sequences ;

    for (int it = 0 ; it < numb_test ; it ++) {

        unsigned char *seq1 = new unsigned char[N];
        unsigned char *seq2 = new unsigned char[N] ; 
        randomSequence(N, seq1) ;
        randomSequence(N, seq2) ;

        unsigned char s1[] = "AAA" ;
        unsigned char s2[] = "AAB" ;

        int scoreSimpleCPU = SmithWatermanScore(seq1, seq2, N, N) ;
        int scoreLazySmithCPU = LazySmith(seq1, seq2, N, N) ;
        int scoreLazySmithParallelThreadsCPU = ParallelLazySmith_threads(seq1, seq2, N, N) ;
        int scoreLazySmithParallelFutureCPU = ParallelLazySmith_futures(seq1, seq2, N, N) ;

        if (scoreSimpleCPU != scoreLazySmithCPU) {
            std::cout << "ERROR TEST " << it << " : " << seq1 << " | " << seq2 << " | score : " << scoreSimpleCPU << " | " << scoreLazySmithCPU << " | " << scoreLazySmithParallelThreadsCPU << " | " << ParallelLazySmith_futures << std::endl ;
        } else {
            std::cout << "SUCCESS TEST " << it << " : " << seq1 << " | " << seq2 << " | score : " << scoreSimpleCPU << " | " << scoreLazySmithCPU << " | " << scoreLazySmithParallelThreadsCPU << " | " << ParallelLazySmith_futures << std::endl ;
        }

    }

}

int main() {


    generateTest(3, 10) ;

    return 0 ;

}