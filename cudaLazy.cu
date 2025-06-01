#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std::chrono;

static const int G_INIT   = 1;
static const int G_EXT    = 1;
static const int MATCH    = 1;
static const int MISMATCH = -1;

__device__ __forceinline__
int dev_score(unsigned char a, unsigned char b) {
    return (a == b) ? MATCH : MISMATCH;
}

__global__
void kernel_updateRow16(
    const unsigned char* __restrict__ d_seq1,
    const unsigned char* __restrict__ d_seq2,
    int* __restrict__ d_H,
    int* __restrict__ d_vE,
    int* __restrict__ d_vF,
    int  n,
    int  m,
    int  i,
    int  segLen
) {
    extern __shared__ int prevRowH_seg[];

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= segLen) return;

    int start = t * 16;
    int width = min(16, n - start);

    for (int x = threadIdx.x; x < width; x += blockDim.x) {
        int globalIdx = (i - 1) * (n + 1) + (start + x);
        prevRowH_seg[t * 16 + x] = d_H[globalIdx];
    }
    __syncthreads();

    int vE = d_vE[t];
    int vF = d_vF[t];
    int bestH_thisRow = 0;

    for (int offset = 0; offset < width; ++offset) {
        int j = start + offset;
        int h_above = prevRowH_seg[t * 16 + offset];
        int s       = dev_score(d_seq1[j], d_seq2[i - 1]);
        int vH      = h_above + s;

        int vHgap = vH - G_INIT;
        int e_new = vE - G_EXT;
        if (vHgap > e_new) e_new = vHgap;
        int f_new = vF - G_EXT;
        if (vHgap > f_new) f_new = vHgap;

        int tmpMax = vH;
        if (vE > tmpMax) tmpMax = vE;
        if (vF > tmpMax) tmpMax = vF;
        if (tmpMax < 0) tmpMax = 0;
        vH = tmpMax;

        d_H[i * (n + 1) + j] = vH;
        if (vH > bestH_thisRow) bestH_thisRow = vH;

        vE = e_new;
        vF = f_new;
    }

    d_vE[t] = vE;
    d_vF[t] = vF;

    if (width > 0) {
        int h0 = d_H[i * (n + 1) + start];
        if (vF > (h0 - G_INIT)) {
            d_H[i * (n + 1) + start] = vF;
        }
    }
}

int LazySmithGPU(const unsigned char* seq1, const unsigned char* seq2, int n, int m) {
    int segLen = (n + 15) / 16;

    unsigned char *d_seq1 = nullptr, *d_seq2 = nullptr;
    cudaMalloc(&d_seq1, n * sizeof(unsigned char));
    cudaMalloc(&d_seq2, m * sizeof(unsigned char));
    cudaMemcpy(d_seq1, seq1, n * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq2, seq2, m * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int *d_H = nullptr;
    cudaMalloc(&d_H, (size_t)(m + 1) * (n + 1) * sizeof(int));
    cudaMemset(d_H, 0, (size_t)(m + 1) * (n + 1) * sizeof(int));

    int *d_vE = nullptr, *d_vF = nullptr;
    cudaMalloc(&d_vE, segLen * sizeof(int));
    cudaMalloc(&d_vF, segLen * sizeof(int));
    cudaMemset(d_vE, 0, segLen * sizeof(int));
    cudaMemset(d_vF, 0, segLen * sizeof(int));

    int threadsPerBlock = 128;
    int blocksPerGrid   = (segLen + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedBytes  = segLen * 16 * sizeof(int);

    for (int i = 1; i <= m; ++i) {
        kernel_updateRow16<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(
            d_seq1, d_seq2,
            d_H,
            d_vE, d_vF,
            n, m,
            i,
            segLen
        );
        cudaDeviceSynchronize();
    }

    std::vector<int> h_lastRow(n + 1);
    cudaMemcpy(h_lastRow.data(),
               d_H + (size_t)m * (n + 1),
               (n + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    int maxScore = 0;
    for (int j = 0; j <= n; ++j) {
        if (h_lastRow[j] > maxScore) {
            maxScore = h_lastRow[j];
        }
    }

    cudaFree(d_seq1);
    cudaFree(d_seq2);
    cudaFree(d_H);
    cudaFree(d_vE);
    cudaFree(d_vF);

    return maxScore;
}

void runTests() {
    struct TestCase {
        std::string seq1, seq2;
        int expected;
    };

    std::vector<TestCase> tests = {
        { "AABDADB", "AADCBAB", 2 },
        { "ABDA",     "ADDB",     1 },
        { "AAA",      "AAA",      3 },
        { "A",        "A",        1 },
        { "A",        "G",        0 },
        {
            "ATCGATCGATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTAGCTAGCTA",
            1
        },
        {
            std::string(400, 'A'),
            std::string(400, 'T'),
            0
        },
        {
            std::string(400, 'A'),
            std::string(400, 'A'),
            400
        },
        {
            std::string(5000, 'A'),
            std::string(5000, 'T'),
            0
        },
        {
            std::string(5000, 'A'),
            std::string(5000, 'A'),
            5000
        }
    };

    std::cout << "Running GPU‐based LazySmith on " << tests.size() << " test cases.\n\n";

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    for (size_t i = 0; i < tests.size(); ++i) {
        const auto& tc = tests[i];
        int n = (int)tc.seq1.size();
        int m = (int)tc.seq2.size();

        // std::cout << "Test " << (i + 1) << ":\n";
        if (n <= 80) {
            std::cout << "  seq1 = \"" << tc.seq1 << "\" (length " << n << ")\n";
        } else {
            std::cout << "  seq1 = \"" << tc.seq1.substr(0, 40) << "...\" (length " << n << ")\n";
        }
        if (m <= 80) {
            std::cout << "  seq2 = \"" << tc.seq2 << "\" (length " << m << ")\n";
        } else {
            std::cout << "  seq2 = \"" << tc.seq2.substr(0, 40) << "...\" (length " << m << ")\n";
        }

        std::vector<unsigned char> h_seq1(n), h_seq2(m);
        for (int j = 0; j < n; ++j) h_seq1[j] = (unsigned char)tc.seq1[j];
        for (int j = 0; j < m; ++j) h_seq2[j] = (unsigned char)tc.seq2[j];

        cudaEventRecord(start_event, 0);
        int gpu_score = LazySmithGPU(h_seq1.data(), h_seq2.data(), n, m);
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);

        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, start_event, stop_event);

        std::cout << "  Expected ≈ " << tc.expected
                  << "   GPU score = " << gpu_score
                  << "   Time = " << gpu_ms << " ms\n\n";
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

int main() {
    kernel_updateRow16<<<1,1,1>>>(nullptr,nullptr,nullptr,nullptr,nullptr,0,0,0,0);
    cudaDeviceSynchronize();

    runTests();
    return 0;
}