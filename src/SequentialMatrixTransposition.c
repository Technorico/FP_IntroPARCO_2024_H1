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
#ifndef RUNS
    #define RUNS 1
#endif
#ifndef RESULT_CSV_PATH
    #define RESULT_CSV_PATH "./results.csv"
#endif


#define RAND_LOB -1000000
#define RAND_UPB 1000000
#define DIV_VALUE 1000.0

#define FIXED_VALUE 32.6

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
    for(uint64_t run_idx = 0; run_idx < RUNS; run_idx++){
        #if DEBUG >= 1
            printf("\n\e[31;1m## RUN %03"PRIu64" ##\e[0m\n", run_idx);
        #endif
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

        #if DEBUG >= 1
            printf("Is matrix Symmetric? -> %s\n", (checkSym(M, SIZE, &wt1, &wt2)?"True":"False"));
        #else
            checkSym(M, SIZE, &wt1, &wt2);
        #endif
        symCheckTime = wt2 - wt1;
        float *T = matTranspose(M, SIZE, &wt1, &wt2);
        #if DEBUG >= 2
            printf("Matrix | Transposed matrix:\n");
            matsPrintSides(M, T, SIZE);
        #endif
        matTransposeTime = wt2 - wt1;

        #if DEBUG >= 0
            FILE *fp = fopen(RESULT_CSV_PATH, "a");
            if(fp == NULL){
                printf("\e[91;1mCould NOT open the file!!\e[0m\n");
            }
            else{
                fprintf(fp, "%"PRIu32":%"PRIu64":%15.9f:%15.9f\n", rand_seed, SIZE, symCheckTime, matTransposeTime);
                fclose(fp);
            }
        #endif
        #if DEBUG >= 1
            printf("The two matrices are equal? -> %s\n", (matCheckEquality(M, M, SIZE)?"True":"False"));
            printf("Time to check the matrix symmetry (omp_get_wtime) = %15.9f sec\n", symCheckTime);
            printf("Time to transpose the matrix (omp_get_wtime)      = %15.9f sec\n", matTransposeTime);
        #endif
        free(T);
        free(M);
    }
    return 0;
}

// ----------------------------------------------------------------------