#include <vector>
#include <random> 
#include <iostream>
#include <cstdlib>
#include <chrono>
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

void generateTest(int N, int numb_test, int mode) {

    double sumLazySmithCPU = 0.0;
    double sumParallelLazySmithThreadsCPU = 0.0;
    bool success = true;

    std::vector<std::pair<std::vector<unsigned char>, std::vector<unsigned char> > > sequences ;

    for (int it = 0 ; it < numb_test ; it ++) {

        unsigned char *seq1 = new unsigned char[N];
        unsigned char *seq2 = new unsigned char[N] ; 
        randomSequence(N, seq1) ;
        randomSequence(N, seq2) ;

        auto t1 = std::chrono::high_resolution_clock::now();
        int scoreSimpleCPU = SmithWatermanScore(seq1, seq2, N, N) ;
        auto t2 = std::chrono::high_resolution_clock::now();
        double timeSimpleCPU = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

        t1 = std::chrono::high_resolution_clock::now();
        int scoreLazySmithCPU = LazySmith(seq1, seq2, N, N) ;
        t2 = std::chrono::high_resolution_clock::now();
        double timeLazySmithCPU = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

        t1 = std::chrono::high_resolution_clock::now();
        int scoreLazySmithParallelThreadsCPU = ParallelLazySmith_threads(seq1, seq2, N, N) ;
        t2 = std::chrono::high_resolution_clock::now();
        double timeLazySmithParallelThreadsCPU = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
        
        //int scoreLazySmithParallelFutureCPU = ParallelLazySmith_futures(seq1, seq2, N, N) ;
        if( mode == 1 ) {  //PER-TEST PRINTING

            double speedupLazySmith = timeSimpleCPU / timeLazySmithCPU;
            double speedupParallel = timeSimpleCPU / timeLazySmithParallelThreadsCPU;
            
            if (scoreSimpleCPU == scoreLazySmithCPU && scoreLazySmithCPU == scoreLazySmithParallelThreadsCPU) {
                std::cout << "TEST " << it << ": score=" << scoreSimpleCPU << "\n"
                  << "SUCCESS" << "\n"
                  << "  SmithWaterman      : " << timeSimpleCPU << " ms\n"
                  << "  LazySmith          : " << timeLazySmithCPU
                  << " ms  (Speedup: " << speedupLazySmith << "x), \n"
                  << "  ParallelLazySmith  : " << timeLazySmithParallelThreadsCPU
                  << " ms  (Speedup: " << speedupParallel << "x), \n"
                  << std::endl;
            } 
            else {
                std::cout << "TEST " << it << ": score=" << scoreSimpleCPU << "\n"
                  << "ERROR" << "\n"
                  << "  SmithWaterman      : " << timeSimpleCPU << " ms\n"
                  << "  LazySmith          : " << timeLazySmithCPU
                  << " ms  (Speedup: " << speedupLazySmith << "x), \n"
                  << "  ParallelLazySmith  : " << timeLazySmithParallelThreadsCPU
                  << " ms  (Speedup: " << speedupParallel << "x), \n"
                  << std::endl;
            }
        }
        else{
            sumLazySmithCPU += timeSimpleCPU / timeLazySmithCPU;
            sumParallelLazySmithThreadsCPU += timeSimpleCPU / timeLazySmithParallelThreadsCPU;
            if (! (scoreSimpleCPU == scoreLazySmithCPU && scoreLazySmithCPU == scoreLazySmithParallelThreadsCPU) ) {
                success = false;
                std::cout<<"ERROR: "<<scoreSimpleCPU<<" | "<<scoreLazySmithCPU<<" | "<<scoreLazySmithParallelThreadsCPU<<"\n";
            }
        }

    }

    if ( mode != 1 ) {
        double averageSpeedupLazySmithCPU = sumLazySmithCPU / numb_test;
        double averageSpeedupParallelLazySmithThreadsCPU = sumParallelLazySmithThreadsCPU / numb_test;

        std::cout << "LENGTH: " << N << ", NUMBER OF TESTS: " << numb_test <<"\n"
                  << "Success: "<< success << "\n"
                  << "  LazySmith          : "
                  << " ms  ( Average Speedup: " << averageSpeedupLazySmithCPU << "x), \n"
                  << "  ParallelLazySmith  : "
                  << " ms  ( Average Speedup: " << averageSpeedupParallelLazySmithThreadsCPU << "x), \n"
                  << std::endl;
    }

}

int main() {

    int mode;
    std::cout << "Select mode:\n";
    std::cout << "  1 - Per-test speedup\n";
    std::cout << "  2 - Average speedup\n";
    std::cout << "Enter choice: ";
    std::cin >> mode;

    //int N = 10; //the length of the sequence
    int num_tests = 10;  //number of tests for each length

    std::vector<int> sequence_lengths = {
        1, 50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
    };

    for( int N : sequence_lengths ) {
        std::cout << "\n============================\n";
        std::cout << "Running tests for sequence length: " << N << "\n";
        std::cout << "============================\n";
        generateTest(N, num_tests, mode);
    }

    //generateTest(N, num_tests, mode) ;

    return 0 ;

}