/*
Description: Sequaential implementation of matrix transposition and check of symmetry
Notes: 
    - It is assumed that all the matrices are square and have the same dimensions
    - A value inside a matrix is identified by the tuple (ROW, COL)
DEBUG DEFINE can assume multiple values(0, 1, 2, 3):
    0. Is the default one, the program will not write on stdout
    1. Write on stdout basic information, like times and equality (M == M)
    2. Write the same of 1, plus it visually displays matrices
    3. Write the same of 2, plus the indexes iteration by iteration of the two main functions
*/

// ------------------------------- DEFINES -------------------------------

#ifndef DEBUG
    #define DEBUG 0
#endif
#ifndef SIZE //The SIZE is in reality the side size of the matrix, so it's like SIZExSIZE 
    #define SIZE 16
#endif
#ifndef RESULT_CSV_PATH
    #define RESULT_CSV_PATH "./results.csv"
#endif
// #define CONST_INIT

#define MAT_TYPE float*
#define MAT_INIT_FUNC matInitSA
#define MAT_PRINT_FUNC matPrintSA
#define MAT_CHECKEQ_FUNC matCheckEqualitySA
#define MAT_FREE_FUNC(mat) freeSA(mat)
//#define MAT_FREE_FUNC(mat) freeMA(mat, SIZE)

#define MAT_TRANSPOSE_FUNC matTransposeImp3SA
#define MAT_CHECKSYM_FUNC checkSymImp3SA
#define MAT_COMP_TRANSPOSE_FUNC matTransposeSA

// ------------------------------- INCLUDES ------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>
#include "utility_functions.c"
#include "base_functions.c"

// --------------------------------- MAIN --------------------------------

int main() {
    uint32_t rand_seed = time(NULL);
    srand(rand_seed);
    MAT_TYPE M = MAT_INIT_FUNC(SIZE);
    #if DEBUG >= 1
        printf("The selected matrix side size is: %lu\n", SIZE);
    #endif
    #if DEBUG >= 2
        printf("Matrix:\n");
        MAT_PRINT_FUNC(M, SIZE);
    #endif

    double wt1, wt2, symCheckTime, matTransposeTime;

    bool isSym = MAT_CHECKSYM_FUNC(M, SIZE, &wt1, &wt2);
    #if DEBUG >= 1
        printf("Is matrix Symmetric? -> %s\n", (isSym?"True":"False"));
    #endif
    symCheckTime = wt2 - wt1;

    MAT_TYPE T = MAT_TRANSPOSE_FUNC(M, SIZE, &wt1, &wt2);
    #if DEBUG >= 2
        printf("Matrix | Transposed matrix:\n");
        MAT_PRINT_FUNC(M, SIZE);
        printf("\n");
        MAT_PRINT_FUNC(T, SIZE);
    #endif
    matTransposeTime = wt2 - wt1;
    #if DEBUG >= 1
        MAT_TYPE TCheck = MAT_COMP_TRANSPOSE_FUNC(M, SIZE, &wt1, &wt2);
    #endif
    #if DEBUG >= 2
        MAT_PRINT_FUNC(TCheck, SIZE);
        printf("\n");
    #endif
    //uint64_t memory_rw = (SIZE * SIZE + SIZE) * 2; //This countes the double writes/reads of diagonal values
    uint64_t memory_rw = (SIZE * SIZE) * 2; //This doesn't take into account the additional memory accesses. (Probably the difference will not be so relevant because of caches)
    //The sequential implementation read one time every source position of the input matrix and writes one time every 
    //destination of the output matrix, except for the diagonals values
    //(those are read and written twice the times respect to the others memory locations).

    //Append to RESULT_CSV_PATH the results of the run
    #if DEBUG >= 0
        saveResults(RESULT_CSV_PATH, rand_seed, SIZE, memory_rw, symCheckTime, matTransposeTime);
    #endif
    #if DEBUG >= 1
        printf("The two matrices are equal? -> %s\n", (MAT_CHECKEQ_FUNC(T, TCheck, SIZE)?"True":"False"));
        printf("Time to check the matrix symmetry (omp_get_wtime) = %15.9f sec\n", symCheckTime);
        printf("Time to transpose the matrix (omp_get_wtime)      = %15.9f sec\n", matTransposeTime);
    #endif
    MAT_FREE_FUNC(T);
    MAT_FREE_FUNC(M);
    #if DEBUG >= 1
        MAT_FREE_FUNC(TCheck);
    #endif
    return 0;
}

// ----------------------------------------------------------------------