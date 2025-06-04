#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <optional>
#include <vector>
#include <iostream>
#include <algorithm>
#include <future>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main() {
    string seq1, seq2;

    seq1 = "";
    seq2 = "";

    for (int i = 0; i < 5000; i++) {
        seq1.append("A");
    }
    for (int i = 0; i < 5000; i++) {
        seq1.append("T");
    }

    cout << seq1 << endl << seq2 << endl;

    return 0;
}
