#include <thread>

// main.cpp
int SmithWatermanScore(unsigned char *seq1, unsigned char *seq2, int n, int m) ;

// lazySmith.cpp
int LazySmith(unsigned char *seq1, unsigned char *seq2, int n, int m) ;

// lazySmith_parallel_threads.cpp
int ParallelLazySmith_threads( unsigned char *seq1, unsigned char *seq2, int n, int m, 
                      int num_threads = std::thread::hardware_concurrency(),
                      int chunkStrategy = 2 ) ;

// lazySmith_parallel_future.cpp
int ParallelLazySmith_futures( unsigned char *seq1, unsigned char *seq2, int n, int m, 
                      int num_threads = std::thread::hardware_concurrency(),
                      int chunkStrategy = 2 ) ;


