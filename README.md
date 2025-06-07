# ConcurrentProject
CSE305 Project: Parallel sequence alignment

**Authors:** Andreea Patarlageanu, Marta Teodora Trales, Joanne Jegou

During this project, we worked on **local** parallel sequence alignement with **affine gap penalty**, with the set of characters `{A, C, G, T}`. We focused on the computation of the score and not the alignement it-self. We implemented the Smith-Waterman algorithm, which is a dynamic programming algorithm. Because of that, parallelization of this algorithm requires some thinking. We first implemented a simple sequential version on CPU, then we introduced the lazy computation of the matrices. We worked on parallelizing those algorithms, and executing them on GPU.

# Organisation of the project and comments

Please note that the commits on Github are not completely representative of the work done by each of the group members, as Andreea and Joanne wrote together some of the files but committed from a single computer.

# Differents files

### CPU
- `main.cpp` is the first sequential implementation of the parallel sequence alignement on CPU
- `lazySmith.cpp` is the sequential implementation of the Lazy Smith Algorithm
- `lazySmith_parallel_threads.cpp` is the parallel version of the Lazy Smith algorithm using threads
- `lazySmith_parallel_future.cpp` is the parallel version of the Lazy Smith algorithm using `future` but it is **NOT WORKING**

### GPU
- `simpleGPU.cu` the simple sequential implementation on GPU
- `cudaSmithM.cu` a second version of the simple sequential implementation on GPU (we had a miscommunication and both of us took that approach on GPU)
- `cudaLazy.cpp` the parallelised lazy implementation of the algorithm on GPU

### Testing
- `Makefile1` the makefile for testing on CPU only (run `make -f Makefile1`)
- `Makefile2` the makefile for testing on CPU and GPU (run `make -f Makefile2`)
- `TestFile.cpp` the testing file for CPU (run `./test_runner1` after compiling)
- `TestFileWithGPU.cpp` the testing file for CPU and GPU (run `./test_runner2` after compiling)
- `algoCPU.h` the header file for .cpp files
- `algoGPU.h` the header file for .cu files 


# Testing

We have two main testing files, which indicates the SUCCESS of the computations (success corresponds to the equality of all computed scores, we get an ERROR otherwise), the time for computations, and the speed-up for each method compared to the simple sequential version). 
_`To test the different files, we commented out all the `void main() functions in the different files. To run the files indepently, you have to uncomment the `main()` functions._

Computations and testing on GPU are executed through SSH on the school's computers. The testing is as follows:
- `TestFile.cpp` is the testing file for CPU computations only. It is compiled through the command `make -f Makefile1` and ran with `./test_runner1`
- `TestFileWithGPU.cpp` is the testing file that includes comparison with GPU computations as well. It is compiled through the command `make -f Makefile2` and ran with `./test_runner2`

The parameters for testing can be changed directly in the `main()` function of the test files: we test the score computation for different values of the length `N` of the sequences to compare, which are defined in the `sequence_lengths` vector. `num_tests` is the number of tests performed for each `N`. We fixed by default the score's parameters `MATCH`, `MISMATCH`, `GAP_INIT` and `GAP_EXT` to respectively 1, -1, 1, 1, and they have to be changed directly in the function files.

When running the test files, the user will be prompted to enter 1 or 2:
- `1` outputs the time and speed-up per test for each `N` (it gives the result of all tests)
- `2` outputs the average speed-up for each method for each `N` (to compare the improvement in performance for each method depending on the sequence length `N`). 
