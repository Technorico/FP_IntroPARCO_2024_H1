#ifndef __UTILITY_FUNCTIONS_C__
    #define __UTILITY_FUNCTIONS_C__

    // ------------------------------- DEFINES -------------------------------

    #ifndef EQ_OFFSET
        #define EQ_OFFSET 0.02
    #endif
    #ifndef FIXED_VALUE
        #define FIXED_VALUE 32.6
    #endif

    #define RAND_LOB -1000000
    #define RAND_UPB 1000000
    #define DIV_VALUE 1000.0

    // #define CONST_INIT

    // ------------------------------- INCLUDES ------------------------------

    #include <stdio.h>
    #include <stdlib.h>
    #include <stdbool.h>
    #include <inttypes.h>

// ---------------------- Matrix Utility Functions ----------------------

    float* matInitSA(uint64_t side_size){
        float *M = (float*)malloc(side_size * side_size * sizeof(float));
        #ifndef CONST_INIT
            for(uint64_t i = 0; i < side_size * side_size; i++){
                M[i] = ((float)(rand() % (RAND_UPB - RAND_LOB + 1) + RAND_LOB)) / DIV_VALUE;
            } //random initialization
        #else
            for(uint64_t i = 0; i < SIZE * SIZE; i++){
                M[i] = FIXED_VALUE;
            } //fixed initialization
        #endif
        return M;
    }

    float** matInitMA(uint64_t side_size){
        float **M = (float**)malloc(side_size * sizeof(float*));
        for(uint64_t i = 0; i < side_size; i++){
            M[i] = (float*)malloc(side_size * sizeof(float));
        }
        #ifndef CONST_INIT
            for(uint64_t c = 0; c < side_size; c++){
                for(uint64_t r = 0; r < side_size; r++){
                    M[c][r] = ((float)(rand() % (RAND_UPB - RAND_LOB + 1) + RAND_LOB)) / DIV_VALUE; 
                }
            } //random initialization
        #else
            for(uint64_t c = 0; c < SIZE; c++){
                for(uint64_t r = 0; r < SIZE; r++){
                    M[c][r] = FIXED_VALUE;
                }
            } //fixed initialization
        #endif
        return M;
    }

    void matPrintSA(float *in_matrix, uint64_t side_size){
        uint64_t total_size = side_size * side_size;
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

    void matPrintMA(float **in_matrix, uint64_t side_size){
        for(uint64_t c = 0; c < side_size; c++){
            for(uint64_t r = 0; r < side_size; r++){
                printf("%10.3f", in_matrix[c][r]);
                printf(" ");
            }
            printf("\n");
        }
    }

    bool matCheckEqualitySA(float *mat1, float *mat2, uint64_t side_size){
        uint64_t total_size = side_size * side_size;
        bool equal = true;
        for(uint64_t i = 0; i < total_size; i++){
            if(mat1[i] >= mat2[i] + EQ_OFFSET || mat1[i] <= mat2[i] - EQ_OFFSET){
                equal = false;
                break;
            }
        }
        return equal;
    }

    bool matCheckEqualityMA(float **mat1, float **mat2, uint64_t side_size){
        bool equal = true;
        for(uint64_t c = 0; c < side_size; c++){
            for(uint64_t r = 0; r < side_size; r++){
                if(mat1[c][r] >= mat2[c][r] + EQ_OFFSET || mat1[c][r] <= mat2[c][r] - EQ_OFFSET){
                    equal = false;
                    break;
                }
            }
        }
        return equal;
    }


    void freeSA(float *mat){
        free(mat);
    }

    void freeMA(float **mat, uint64_t side_size){
        for(uint64_t i = 0; i < side_size; i++){
            free(mat[i]);
        }
        free(mat);
    }

    // -------------------------- Utility Functions -------------------------

    void saveResults(char *file_path, uint32_t rand_seed, uint64_t side_size, uint64_t memory_rw, uint64_t n_of_threads, float symCheckTime, float matTransposeTime){
        FILE *fp = fopen(file_path, "a");
        if(fp == NULL){
            printf("\e[91;1mCould NOT open the file!!\e[0m\n");
        }
        else{
            fprintf(fp, "%"PRIu32":%"PRIu64":%"PRIu64":%"PRIu64":%015.9f:%015.9f\n", rand_seed, side_size, memory_rw, n_of_threads, symCheckTime, matTransposeTime);
            fclose(fp);
        }
    }

    char* filename_from_path(char *path){
        uint64_t index = 0;
        uint64_t start_name_index = 0;
        char ch = path[index];
        while(ch != '\0'){
            if((ch == '/' || ch == '\\') && path[index + 1] != '\0'){
                start_name_index = index + 1;
            }
            index++;
            ch = path[index];
        }
        uint64_t name_str_lenght = (index + 1 - start_name_index);
        char *filename = (char *)malloc(name_str_lenght * sizeof(char));
        for(uint64_t i = 0; i < name_str_lenght; i++){
            filename[i] = path[i + start_name_index];
        }
        return filename;
    }
    
    // ----------------------------------------------------------------------

#endif