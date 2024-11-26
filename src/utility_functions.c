#ifndef __UTILITY_FUNCTIONS_C__
    #define __UTILITY_FUNCTIONS_C__

    #ifndef DEBUG
        #define DEBUG 0
    #endif

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

    // ----------------------------------------------------------------------
#endif