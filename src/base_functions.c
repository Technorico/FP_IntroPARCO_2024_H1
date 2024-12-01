#ifndef __BASE_FUNCTIONS_C__
    #define __BASE_FUNCTIONS_C__

    // ------------------------------- DEFINES -------------------------------

    #ifndef DEBUG
        #define DEBUG 0
    #endif

    // ------------------------------- INCLUDES ------------------------------

    #include <stdio.h>
    #include <stdlib.h>
    #include <inttypes.h>
    #include <stdbool.h>
    #ifdef _OPENMP
        #include <omp.h>
    #else
        #error "You must use '-fopenmp' compilation flag to compile this code."
    #endif

    // ---------------------- Function Implementations ----------------------

    bool checkSymSA(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        bool isSym = true;
        *wt_start = omp_get_wtime();
        for(uint64_t c_idx = 0; c_idx < side_size; c_idx++){
            for(uint64_t r_idx = 0; r_idx < side_size; r_idx++){
                if(in_matrix[c_idx * side_size + r_idx] != in_matrix[r_idx * side_size + c_idx])
                    isSym = false;
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", c_idx, r_idx, c_idx * side_size + r_idx, r_idx, c_idx, r_idx * side_size + c_idx);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return isSym;
    }

    float* matTransposeSA(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        float* temp_mat = (float*)malloc(side_size * side_size * sizeof(float));
        *wt_start = omp_get_wtime();
        for(uint64_t c_idx = 0; c_idx < side_size; c_idx++){
            for(uint64_t r_idx = 0; r_idx < side_size; r_idx++){
                temp_mat[r_idx * side_size + c_idx] = in_matrix[c_idx * side_size + r_idx];
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", c_idx, r_idx, c_idx * side_size + r_idx, r_idx, c_idx, r_idx * side_size + c_idx);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return temp_mat;
    }

    bool checkSymMA(float **in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        bool isSym = true;
        *wt_start = omp_get_wtime();
        for(uint64_t c_idx = 0; c_idx < side_size; c_idx++){
            for(uint64_t r_idx = 0; r_idx < side_size; r_idx++){
                if(in_matrix[c_idx][r_idx] != in_matrix[r_idx][c_idx])
                    isSym = false;
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64")\n", c_idx, r_idx, r_idx, c_idx);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return isSym;
    }

    float** matTransposeMA(float **in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        float **temp_mat = (float**)malloc(side_size * sizeof(float*));
        for(uint64_t i = 0; i < side_size; i++){
            temp_mat[i] = (float*)malloc(side_size * sizeof(float));
        }
        *wt_start = omp_get_wtime();
        for(uint64_t c_idx = 0; c_idx < side_size; c_idx++){
            for(uint64_t r_idx = 0; r_idx < side_size; r_idx++){
                temp_mat[c_idx][r_idx] = in_matrix[r_idx][c_idx];
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64")\n", c_idx, r_idx, r_idx, c_idx);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return temp_mat;
    }

#endif
