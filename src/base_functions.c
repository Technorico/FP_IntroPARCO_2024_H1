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
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
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
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
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
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return isSym;
    }

    float* matTransposeMA(float **in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        float** temp_mat = (float*)malloc(side_size * side_size * sizeof(float));
        for(uint64_t i = 0; i < side_size; i++){
            temp_mat[i] = (float*)malloc(side_size * sizeof(float));
        }
        *wt_start = omp_get_wtime();
        for(uint64_t c_idx = 0; c_idx < side_size; c_idx++){
            for(uint64_t r_idx = 0; r_idx < side_size; r_idx++){
                temp_mat[c_idx][r_idx] = in_matrix[r_idx][c_idx];
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
    bool checkSymImp1SA(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        bool isSym = true;
        *wt_start = omp_get_wtime();
        for(uint64_t c_idx = 0; c_idx < side_size; c_idx++){
            for(uint64_t r_idx = 0; r_idx < side_size; r_idx++){
                if(in_matrix[c_idx * side_size + r_idx] != in_matrix[r_idx * side_size + c_idx])
                    isSym = false;
                    __builtin_prefetch(&(in_matrix[c_idx * side_size + r_idx + PREFETCH_OFFSET]), 0, 3);
                    __builtin_prefetch(&(in_matrix[r_idx * side_size + c_idx + PREFETCH_OFFSET]), 0, 3);
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return isSym;
    }

    float* matTransposeImp1SA(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        float* temp_mat = (float*)malloc(side_size * side_size * sizeof(float));
        *wt_start = omp_get_wtime();
        for(uint64_t c_idx = 0; c_idx < side_size; c_idx++){
            for(uint64_t r_idx = 0; r_idx < side_size; r_idx++){
                temp_mat[r_idx * side_size + c_idx] = in_matrix[c_idx * side_size + r_idx];
                __builtin_prefetch(&(in_matrix[c_idx * side_size + r_idx + PREFETCH_OFFSET]), 0, 3);
                __builtin_prefetch(&(temp_mat[r_idx * side_size + c_idx + PREFETCH_OFFSET]), 1, 3);
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return temp_mat;
    }

    bool checkSymImp1MA(float **in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        bool isSym = true;
        *wt_start = omp_get_wtime();
        for(uint64_t c_idx = 0; c_idx < side_size; c_idx++){
            for(uint64_t r_idx = 0; r_idx < side_size; r_idx++){
                if(in_matrix[c_idx][r_idx] != in_matrix[r_idx][c_idx])
                    isSym = false;
                    __builtin_prefetch(&(in_matrix[c_idx][r_idx + PREFETCH_OFFSET]), 0, 3);
                    __builtin_prefetch(&(in_matrix[c_idx + PREFETCH_OFFSET][r_idx]), 0, 3);
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return isSym;
    }

    float* matTransposeImp1MA(float **in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        float** temp_mat = (float*)malloc(side_size * side_size * sizeof(float));
        for(uint64_t i = 0; i < side_size; i++){
            temp_mat[i] = (float*)malloc(side_size * sizeof(float));
        }
        *wt_start = omp_get_wtime();
        for(uint64_t c_idx = 0; c_idx < side_size; c_idx++){
            for(uint64_t r_idx = 0; r_idx < side_size; r_idx++){
                temp_mat[c_idx][r_idx] = in_matrix[r_idx][c_idx];
                __builtin_prefetch(&(in_matrix[c_idx][r_idx + PREFETCH_OFFSET]), 0, 3);
                __builtin_prefetch(&(temp_mat[c_idx + PREFETCH_OFFSET][r_idx]), 1, 3);
                #if DEBUG >= 3
                    printf("ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64") <|> ELEM(%"PRIu64", %"PRIu64") REAL-IDX(%"PRIu64")\n", margin_idx, check_idx - margin_idx, side_size * check_idx + margin_idx, check_idx - margin_idx, margin_idx, check_idx + margin_idx * side_size);
                #endif
            }
        }
        *wt_end = omp_get_wtime();
        return temp_mat;
    }

    // Implemented checkSymImp2 and matTransposeImp2 with "tiling/blocking" as implicit parallelization/optimization technique
    bool checkSymImp2SA(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
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

    float* matTransposeImp2SA(float *in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
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
                    for(uint64_t inBlk_R_idx = inBlk_C_idx; inBlk_R_idx < TILE_SIDE_SIZE; inBlk_R_idx++){
                        #if DEBUG >= 3
                            printf("Block C|R - InBlock C|R: %"PRIu64"|%"PRIu64" - %"PRIu64"|%"PRIu64"\n", Cblk_idx, Rblk_idx, inBlk_C_idx, inBlk_R_idx);
                        #endif
                        //Read index calculated as: Rblk_idx * TILE_SIDE_SIZE + inBlk_R_idx + Cblk_idx * elems_in_row_of_blks + inBlk_C_idx * side_size
                        //Write index calculated as: Cblk_idx * TILE_SIDE_SIZE + inBlk_C_idx + Rblk_idx * elems_in_row_of_blks + inBlk_R_idx * side_size
                        temp_mat[Cblk_idx * TILE_SIDE_SIZE + inBlk_C_idx + Rblk_idx * elems_in_row_of_blks + inBlk_R_idx * side_size] = in_matrix[Rblk_idx * TILE_SIDE_SIZE + inBlk_R_idx + Cblk_idx * elems_in_row_of_blks + inBlk_C_idx * side_size];
                        temp_mat[Rblk_idx * TILE_SIDE_SIZE + inBlk_R_idx + Cblk_idx * elems_in_row_of_blks + inBlk_C_idx * side_size] = in_matrix[Cblk_idx * TILE_SIDE_SIZE + inBlk_C_idx + Rblk_idx * elems_in_row_of_blks + inBlk_R_idx * side_size];
                    }
                }
            }
        }
        *wt_end = omp_get_wtime();
        return temp_mat;
    }

    bool checkSymImp2MA(float **in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        bool isSym = true;
        *wt_start = omp_get_wtime();
        uint64_t n_side_blocks = side_size / TILE_SIDE_SIZE;
        uint64_t elems_in_row_of_blks = n_side_blocks * TILE_SIDE_SIZE * TILE_SIDE_SIZE;
        for(uint64_t Cblk_idx = 0; Cblk_idx < side_size; Cblk_idx+=TILE_SIDE_SIZE){   
            for(uint64_t Rblk_idx = 0; Rblk_idx < side_size; Rblk_idx+=TILE_SIDE_SIZE){
                for(uint64_t inBlk_C_idx = 0; inBlk_C_idx < TILE_SIDE_SIZE; inBlk_C_idx++){
                    for(uint64_t inBlk_R_idx = 0; inBlk_R_idx < TILE_SIDE_SIZE; inBlk_R_idx++){
                        #if DEBUG >= 3
                            printf("Block C|R - InBlock C|R: %"PRIu64"|%"PRIu64" - %"PRIu64"|%"PRIu64"\n", Cblk_idx, Rblk_idx, inBlk_C_idx, inBlk_R_idx);
                        #endif
                        if(in_matrix[Rblk_idx + inBlk_R_idx][Cblk_idx + inBlk_C_idx] != in_matrix[Cblk_idx + inBlk_C_idx][Rblk_idx + inBlk_R_idx])
                            isSym = false;
                    }
                }
            }
        }
        *wt_end = omp_get_wtime();
        return isSym;
    }

    float** matTransposeImp2MA(float **in_matrix, uint64_t side_size, double *wt_start, double *wt_end){
        #if DEBUG >= 1
            printf("Function: %s\n", __func__);
        #endif
        float **temp_mat = (float**)malloc(side_size * sizeof(float*));
        for(uint64_t i = 0; i < side_size; i++){
            temp_mat[i] = (float*)malloc(side_size * sizeof(float));
        }
        *wt_start = omp_get_wtime();
        uint64_t n_side_blocks = side_size / TILE_SIDE_SIZE;
        uint64_t elems_in_row_of_blks = n_side_blocks * TILE_SIDE_SIZE * TILE_SIDE_SIZE;
        for(uint64_t Cblk_idx = 0; Cblk_idx < side_size; Cblk_idx+=TILE_SIDE_SIZE){   
            for(uint64_t Rblk_idx = 0; Rblk_idx < side_size; Rblk_idx+=TILE_SIDE_SIZE){
                for(uint64_t inBlk_C_idx = 0; inBlk_C_idx < TILE_SIDE_SIZE; inBlk_C_idx++){
                    for(uint64_t inBlk_R_idx = 0; inBlk_R_idx < TILE_SIDE_SIZE; inBlk_R_idx++){
                        #if DEBUG >= 3
                            printf("Block C|R - InBlock C|R: %"PRIu64"|%"PRIu64" - %"PRIu64"|%"PRIu64"\n", Cblk_idx, Rblk_idx, inBlk_C_idx, inBlk_R_idx);
                        #endif
                        temp_mat[Rblk_idx + inBlk_R_idx][Cblk_idx + inBlk_C_idx] = in_matrix[Cblk_idx + inBlk_C_idx][Rblk_idx + inBlk_R_idx];
                        //temp_mat[inBlk_C_idx * side_size + Cblk_idx * elems_in_row_of_blks + Rblk_idx * TILE_SIDE_SIZE + inBlk_R_idx] = in_matrix[inBlk_R_idx * side_size + Rblk_idx * elems_in_row_of_blks + Cblk_idx * TILE_SIDE_SIZE + inBlk_C_idx];
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
