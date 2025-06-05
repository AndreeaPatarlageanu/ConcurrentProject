# ConcurrentProject
CSE305 Project: Parallel sequence alignment

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
