#ifndef __BASE_FUNCTIONS_C__
    #define __BASE_FUNCTIONS_C__

    // ------------------------------- DEFINES -------------------------------

    #ifndef DEBUG
        #define DEBUG 0
    #endif
    #ifndef PREFETCH_OFFSET
        #define PREFETCH_OFFSET 4
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

    bool checkSym(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
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
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
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

    // Implemented checkSymImp1 and matTransposeImp1 with "__builtin_prefetch()" as implicit parallelization/optimization technique
    // __builtin_prefetch(mem_pointer, rw, locality) -> read0 - write1 | locality0-3
    bool checkSymImp1(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        bool isSym = true;
        *wt_start = omp_get_wtime();
        __builtin_prefetch(&(in_matrix[SIZE]), 0, 1);
        __builtin_prefetch(&(in_matrix[0]), 0, 1);
        for(uint64_t margin_idx = 0; margin_idx < side_size; margin_idx++){
            // Setting the inital value to 1 instead of 0: in this way the diagonal values are skipped
            // This should not be done in this function, because this will imply the diagonal values are not copied!
            for(uint64_t check_idx = (1 + margin_idx); check_idx < side_size; check_idx++){
                if(in_matrix[side_size * check_idx + margin_idx] != in_matrix[check_idx + margin_idx * side_size])
                    isSym = false;
                __builtin_prefetch(&(in_matrix[side_size * (check_idx + PREFETCH_OFFSET) + margin_idx]), 0, 3);
                __builtin_prefetch(&(in_matrix[check_idx + PREFETCH_OFFSET + margin_idx * side_size]), 0, 3);
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return isSym;
    }

    float* matTransposeImp1(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
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
                __builtin_prefetch(&(in_matrix[side_size * (check_idx + PREFETCH_OFFSET) + margin_idx]), 0, 3);
                __builtin_prefetch(&(in_matrix[check_idx + PREFETCH_OFFSET + margin_idx * side_size]), 0, 3);
                __builtin_prefetch(&(temp_mat[side_size * (check_idx + PREFETCH_OFFSET) + margin_idx]), 1, 3);
                __builtin_prefetch(&(temp_mat[check_idx + PREFETCH_OFFSET + margin_idx * side_size]), 1, 3);
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return temp_mat;
    }

    // Implemented checkSymImp2 and matTransposeImp2 with "tiling/blocking" as implicit parallelization/optimization technique
    
    // Implemented checkSymImp and matTransposeImp with "manual partial unroll" as implicit parallelization/optimization technique

    // Implemented checkSymImp and matTransposeImp with "__builtin_expect" as implicit parallelization/optimization technique
#endif
