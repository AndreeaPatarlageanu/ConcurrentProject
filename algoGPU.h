#ifdef __cplusplus
extern "C" {
#endif

int SequentialSmithWatermanScoreGPU(unsigned char* seq1, unsigned char* seq2, int len1, int len2);

int SmithWatermanLazyGPU(const unsigned char* seq1, const unsigned char* seq2, int n, int m) ;

int SmithWatermanScoreCUDA(const unsigned char* seq1, const unsigned char* seq2, int n, int m) ;


#ifdef __cplusplus
}
#endif