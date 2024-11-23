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

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#include <omp.h>

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
// -----------------------------------------------------------------------

bool checkSym(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end);

float* matTranspose(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end);

void matPrint(float *in_matrix, uint64_t side_size);

void matsPrintSides(float *mat1, float *mat2, uint64_t side_size);

bool matCheckEquality(float *mat1, float *mat2, uint64_t side_size);

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
    }
    return 0;
}

// ---------------------- Function Implementations ----------------------

bool checkSym(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
    bool isSym = true;
    *wt_start = omp_get_wtime();
    for(uint64_t margin_idx = 0; margin_idx < side_size; margin_idx++){
        // Setting the inital value to 1 instead of 0: in this way the diagonal values are skipped
        // This should not be done in this function, because this will imply the diagonal values are not copied!
        for(uint64_t check_idx = (1 + margin_idx); check_idx < side_size; check_idx++){
            if(in_matrix[side_size * check_idx + margin_idx] != in_matrix[check_idx + margin_idx * side_size])
                isSym = false;
            #if DEBUG >= 3
                printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
            #endif
        }
    }
    *wt_end = omp_get_wtime();
    return isSym;
}

float* matTranspose(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
    float* temp_mat = (float*)malloc(side_size * side_size * sizeof(float));
    *wt_start = omp_get_wtime();
    for(uint64_t margin_idx = 0; margin_idx < side_size; margin_idx++){
        // Setting the inital value to 1 instead of 0: in this way the diagonal values are skipped
        // This should not be done in this function, because this will imply the diagonal values are not copied!
        for(uint64_t check_idx = (0 + margin_idx); check_idx < side_size; check_idx++){
            // Assign the values at (margin_idx, [0,1,2,3,...]) to ([0,1,2,3,...], margin_idx)
            temp_mat[side_size * check_idx + margin_idx] = in_matrix[check_idx + margin_idx * side_size];
            // Assign the values at ([0,1,2,3,...], margin_idx) to (margin_idx, [0,1,2,3,...])
            temp_mat[check_idx + margin_idx * side_size] = in_matrix[side_size * check_idx + margin_idx];
            #if DEBUG >= 3
                printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
            #endif
        }
    }
    *wt_end = omp_get_wtime();
    return temp_mat;
}

void matPrint(float *in_matrix, uint64_t side_size){
    uint64_t total_size = side_size*side_size;
    for(uint64_t i = 0; i < total_size; i++){
        printf("%10.3f", in_matrix[i]);
        if(0 == (i + 1) % side_size){
            printf("\n");
        }
        else{
            printf(" ");
        }
    }
}

void matsPrintSides(float *mat1, float *mat2, uint64_t side_size){
    uint64_t total_size = side_size*side_size;
    for(uint64_t i = 0; i < total_size; i++){
        printf("%10.3f", mat1[i]);
        if(0 == (i + 1) % side_size){
            printf(" |");
            for(uint64_t j = i + 1 - side_size; j < i + 1; j++){
                printf(" %10.3f", mat2[j]);
            }
            printf("\n");
        }
        else{
            printf(" ");
        }
    }
}

bool matCheckEquality(float *mat1, float *mat2, uint64_t side_size){
    uint64_t total_size = side_size*side_size;
    bool equal = true;
    for(uint64_t i = 0; i < total_size; i++){
        if(mat1[i] != mat2[i]){
            equal = false;
            break;
        }
    }
    return equal;
}

// ----------------------------------------------------------------------