#ifndef __UTILITY_FUNCTIONS_C__
    #define __UTILITY_FUNCTIONS_C__

    // ------------------------------- DEFINES -------------------------------

    #ifndef DEBUG
        #define DEBUG 0
    #endif
    #ifndef RESULT_CSV_PATH
        #define RESULT_CSV_PATH "./results.csv"
    #endif

    // ------------------------------- INCLUDES ------------------------------

    #include <stdio.h>
    #include <stdlib.h>
    #include <stdbool.h>
    #include <inttypes.h>

    // -------------------------- Utility Functions -------------------------

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

    void saveResults(char *file_path, uint32_t rand_seed, uint64_t side_size, uint64_t memory_rw, float symCheckTime, float matTransposeTime){
        FILE *fp = fopen(file_path, "a");
        if(fp == NULL){
            printf("\e[91;1mCould NOT open the file!!\e[0m\n");
        }
        else{
            fprintf(fp, "%"PRIu32":%"PRIu64":%"PRIu64":%015.9f:%015.9f\n", rand_seed, side_size, memory_rw, symCheckTime, matTransposeTime);
            fclose(fp);
        }
    }

    // ----------------------------------------------------------------------
#endif