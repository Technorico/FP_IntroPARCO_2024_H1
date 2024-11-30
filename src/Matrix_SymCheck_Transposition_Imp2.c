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
    #define SIZE 5
#endif
#ifndef RESULT_CSV_PATH
    #define RESULT_CSV_PATH "./results.csv"
#endif

#define RAND_LOB -1000000
#define RAND_UPB 1000000
#define DIV_VALUE 1000.0

#define FIXED_VALUE 32.6

#define MAT_TRANSPOSE_FUNC matTransposeImp2
#define CHECK_SYM_FUNC checkSymImp2

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
    float *M = (float*)malloc(SIZE * SIZE * sizeof(float));
    #ifndef CONST_INIT
        for(uint64_t i = 0; i < SIZE * SIZE; i++){ M[i] = ((float)(rand() % (RAND_UPB - RAND_LOB + 1) + RAND_LOB)) / DIV_VALUE; } //random initialization
    #else
        for(uint64_t i = 0; i < SIZE * SIZE; i++){ M[i] = FIXED_VALUE; } //fixed initialization
    #endif
    #if DEBUG >= 1
        printf("The selected matrix side size is: %lu\n", SIZE);
    #endif
    #if DEBUG >= 2
        printf("Matrix:\n");
        matPrint(M, SIZE);
    #endif

    double wt1, wt2, symCheckTime, matTransposeTime;

    bool isSym = CHECK_SYM_FUNC(M, SIZE, &wt1, &wt2);
    #if DEBUG >= 1
        printf("Is matrix Symmetric? -> %s\n", (isSym?"True":"False"));
    #endif
    symCheckTime = wt2 - wt1;

    float *T = MAT_TRANSPOSE_FUNC(M, SIZE, &wt1, &wt2);
    #if DEBUG >= 2
        printf("Matrix | Transposed matrix:\n");
        matPrint(M, SIZE);
        printf("\n");
        matPrint(T, SIZE);
        //matsPrintSides(M, T, SIZE);
    #endif
    matTransposeTime = wt2 - wt1;
    #if DEBUG >= 1
        float *TCheck = matTranspose(M, SIZE, &wt1, &wt2);
    #endif
    #if DEBUG >= 2
        matPrint(TCheck, SIZE);
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
        printf("The two matrices are equal? -> %s\n", (matCheckEquality(T, TCheck, SIZE)?"True":"False"));
        printf("Time to check the matrix symmetry (omp_get_wtime) = %15.9f sec\n", symCheckTime);
        printf("Time to transpose the matrix (omp_get_wtime)      = %15.9f sec\n", matTransposeTime);
    #endif
    free(T);
    free(M);
    #if DEBUG >= 1
        free(TCheck);
    #endif
    return 0;
}

// ----------------------------------------------------------------------