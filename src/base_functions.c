#ifndef __BASE_FUNCTIONS_C__
    #define __BASE_FUNCTIONS_C__

    // ------------------------------- DEFINES -------------------------------

    #ifndef DEBUG
        #define DEBUG 0
    #endif
    #ifndef PREFETCH_OFFSET
        #define PREFETCH_OFFSET 5
    #endif
    #ifndef TILE_SIDE_SIZE
        #define TILE_SIDE_SIZE 16
        //This is the optimal size for having a tile row loaded within a single cache line
        //cache_line_size_in_bytes / sizeof(float) = TILE_TOTAL_SIZE | 64 / 4 = 16 elements
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
        //__builtin_prefetch(&(in_matrix[side_size]), 0, 1);
        //__builtin_prefetch(&(in_matrix[0]), 0, 1);
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
    bool checkSymImp2(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        bool isSym = true;
        *wt_start = omp_get_wtime();
        uint64_t n_side_blocks = side_size / TILE_SIDE_SIZE;
        uint64_t elems_in_row_of_blks = n_side_blocks * TILE_SIDE_SIZE * TILE_SIDE_SIZE;
        for(uint64_t Cblk_idx = 0; Cblk_idx < n_side_blocks; Cblk_idx++){   
            for(uint64_t Rblk_idx = 0; Rblk_idx < n_side_blocks; Rblk_idx++){
                for(uint64_t inBlk_C_idx = 0; inBlk_C_idx < TILE_SIDE_SIZE; inBlk_C_idx++){
                    for(uint64_t inBlk_R_idx = 0; inBlk_R_idx < TILE_SIDE_SIZE; inBlk_R_idx++){
                        #if DEBUG >= 3
                            printf("Block C|R - InBlock C|R: %"PRIu64"|%"PRIu64" - %"PRIu64"|%"PRIu64"\n", Cblk_idx, Rblk_idx, inBlk_C_idx, inBlk_R_idx);
                        #endif
                        //Read index calculated as: Rblk_idx * TILE_SIDE_SIZE + inBlk_R_idx + Cblk_idx * elems_in_row_of_blks + inBlk_C_idx * side_size
                        //Read transposed index calculated as: Cblk_idx * TILE_SIDE_SIZE + inBlk_C_idx + Rblk_idx * elems_in_row_of_blks + inBlk_R_idx * side_size
                        if(in_matrix[Cblk_idx * TILE_SIDE_SIZE + inBlk_C_idx + Rblk_idx * elems_in_row_of_blks + inBlk_R_idx * side_size] != in_matrix[Rblk_idx * TILE_SIDE_SIZE + inBlk_R_idx + Cblk_idx * elems_in_row_of_blks + inBlk_C_idx * side_size])
                            isSym = false;
                    }
                }
            }
        }
        *wt_end = omp_get_wtime();
        return isSym;
    }

    float* matTransposeImp2(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        float* temp_mat = (float*)malloc(side_size * side_size * sizeof(float));
        *wt_start = omp_get_wtime();
        uint64_t n_side_blocks = side_size / TILE_SIDE_SIZE;
        uint64_t elems_in_row_of_blks = n_side_blocks * TILE_SIDE_SIZE * TILE_SIDE_SIZE;
        for(uint64_t Cblk_idx = 0; Cblk_idx < n_side_blocks; Cblk_idx++){   
            for(uint64_t Rblk_idx = 0; Rblk_idx < n_side_blocks; Rblk_idx++){
                for(uint64_t inBlk_C_idx = 0; inBlk_C_idx < TILE_SIDE_SIZE; inBlk_C_idx++){
                    for(uint64_t inBlk_R_idx = 0; inBlk_R_idx < TILE_SIDE_SIZE; inBlk_R_idx++){
                        #if DEBUG >= 3
                            printf("Block C|R - InBlock C|R: %"PRIu64"|%"PRIu64" - %"PRIu64"|%"PRIu64"\n", Cblk_idx, Rblk_idx, inBlk_C_idx, inBlk_R_idx);
                        #endif
                        //Read index calculated as: Rblk_idx * TILE_SIDE_SIZE + inBlk_R_idx + Cblk_idx * elems_in_row_of_blks + inBlk_C_idx * side_size
                        //Write index calculated as: Cblk_idx * TILE_SIDE_SIZE + inBlk_C_idx + Rblk_idx * elems_in_row_of_blks + inBlk_R_idx * side_size
                        temp_mat[Cblk_idx * TILE_SIDE_SIZE + inBlk_C_idx + Rblk_idx * elems_in_row_of_blks + inBlk_R_idx * side_size] = in_matrix[Rblk_idx * TILE_SIDE_SIZE + inBlk_R_idx + Cblk_idx * elems_in_row_of_blks + inBlk_C_idx * side_size];
                    }
                }
            }
        }
        *wt_end = omp_get_wtime();
        return temp_mat;
    }

    // Implemented checkSymImp and matTransposeImp with "manual partial unroll" as implicit parallelization/optimization technique

    // Implemented checkSymImp and matTransposeImp with "__builtin_expect" as implicit parallelization/optimization technique
#endif
