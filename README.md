# IntroPARCO 2024 H1

### Federico Peruzzo

#### _First deliverable homework of Introduction to Parallel Computing course of University of Trento._

---

## Specificantions

This repository contains the code used for the first homework.
Specs:
- The project's codes are supposed to work with square matrices (*SIZE* identifies the side size of the matrix)
- For best compatibility the codes requires (or code would not compile):
  - *SIZE* must be a multiple of *TILE_SIDE_SIZE*;
  - *SIZE* must be a multiple of *UNROLL4_INCREMENT*;
  - *TILE_SIDE_SIZE* must be a multiple of *UNROLL4_INCREMENT*.
- Because of the use of OpenMP for some implementations and for getting the time reference of the kernel functions, it is mandatory to compile code with *\-fopenmp*

The codes are divided in:
- basic naive implementations for **symmetry check** and **transposition**, without any optimization and/or framework;
- implementations with ILP(Instruction Level Parallelism) or other implicit parallelization techniques;
- implementations with OpenMP for exploiting multicore systems/machines

> Every implementation has been coded to have a **SA**(or SingleArray) and **MA**(or MultiArray) versions.

_**SA** and **MA** refer to how the matrix is stored in memory(through a single contiguous array of floats or an array of arrays of floats)_

## Table of Implementations

Currently, these are the implementations available in the repository
| Name                                 | Type of Implementation          | Description                                                                                                                                                            |
| :----------------------------------- | :-----------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Matrix_SymCheck_Transposition        | Naive Sequential                | Matrices are accessed element by element, the side size of the matrix is defined by *SIZE*                                                                             |
| Matrix_SymCheck_TranspositionImp1    | Seq + prefetch                  | Matrices are accessed element by element, prefetching elements *PREFETCH_OFFSET* iterations before it is used                                                          |
| Matrix_SymCheck_TranspositionImp2    | Seq + Tiling                    | Matrices are accessed block by block, and inside element by element. A block side size is defined by *TILE_SIDE_SIZE*                                                  |
| Matrix_SymCheck_TranspositionImp3    | Seq + Unroll4                   | Matrices are accessed 4 elements at a time(for what concerns the row index), this reduces by 4 times the innermost loop (normally the one related to the row index)    |
| Matrix_SymCheck_TranspositionImp4    | Seq + Branch Expect Directive   | Matrices are accessed element by element, but the branches inside the main functions are treated like it is expected the more probable branch is always taken          |
| Matrix_SymCheck_TranspositionImp5    | Seq + Tiling + Unroll4          | Matrices are accessed block by block, and inside 4 elements at a time(for what concerns the row index)                                                                 |
| Matrix_SymCheck_TranspositionExp1    | OpenMP + Collapse2              | Matrices are accessed in parallel by the number of threads specified with the *OMP_NUM_THREADS* env var, before running the compiled code                              |
| Matrix_SymCheck_TranspositionExp2    | OpenMP + Tiling + Collapse2     | Matrices are accessed in parallel, exploiting the tiling technique; thanks to the collapse2, the outermost fors are unrolled and each block is assigned to a thread    |
| Matrix_SymCheck_TranspositionExp3    | OpenMP + Tiling + Collapse4     | Matrices are accessed in parallel, exploiting the tiling technique; with collapse4, all fors are unrolled and the pool of operations is split among threads            |

## What can be configured?

> All the changes to the default configurations can be made using *"**\-D**"* flag during compilation.

_I decided to use this way to make things configurable, because of this approach's simplicity and functionality._
_In this way, *DEBUG* settings and *SIZE* become known at compile time, simplifying the compiler's job and avoiding losing time on conditional prints or whatever depends on static user decisions._

Currently, these are the important DEFINES/MACRO inside the project
| Name                | Default&nbsp;Value   | Modifiability | Files                                      | Description&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                                                                    |
| :------------------ | :------------------: | :-----------: | :----------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------- |
| SIZE                | 16                   | COMPILATION   | Matrix_SymCheck_Transposition\*.c<br>exp_functions.c<br>imp_functions.c<br> | Defines the side size of the matrix. Total size is given by _SIZE \* SIZE_                    |
| DEBUG               | 1                    | COMPILATION   | !utility_functions.c                       | Defines the level of debug verbosity of the program                                                                            |
| PREFETCH_OFFSET     | 5                    | COMPILATION   | imp_functions.c                            | Defines the index offset from the current element, of the element that the prefetch instruction has to load                    |
| TILE_SIDE_SIZE      | 16                   | COMPILATION   | exp_functions.c<br>imp_functions.c         | Defines the side size of block, used for the tiling technique                                                                  |
| FIXED_VALUE         | 32.6                 | COMPILATION   | utility_functions.c                        | Defines the value used for initialization of matrix in fixed init mode                                                         |
| EQ_OFFSET           | 0.02                 | COMPILATION   | utility_functions.c                        | Defines what is the tollerance when comparing the floats of two matrices                                                       |
| RESULT_CSV_PATH     | ./results.csv        | COMPILATION   | Matrix_SymCheck_Transposition\*.c          | Defines the path where to save the results file (the path is relative to where the code is ran)                                |
| UNROLL4_INCREMENT   | 4                    | CODE          | utility_functions.c                        | Defines the increment to feed to the for when using unrolling4                                                                 |
| RAND_LOB            | -1000000             | CODE          | utility_functions.c                        | Defines the LOwerBound of the random function                                                                                  |
| RAND_UPB            | 1000000              | CODE          | utility_functions.c                        | Defines the UPperBound of the random function                                                                                  |
| DIV_VALUE           | 1000.0               | CODE          | utility_functions.c                        | Defines the division factor to apply to every number computed by the random function                                           |
| CONST_INIT          | **NOT DEFINED**      | COMPILATION   | utility_functions.c                        | When defined it tells the matrix initialization function to use the *FIXED_VALUE* as constant value for initialization         |

DEBUG macro can assume multiple values(0, 1, 2, 3):

0. The program will not write on stdout. (Results are written to *RESULTS_CSV_PATH*)
1. Is the default one, writes on stdout basic and important information, like times and equality _(M == T)_
2. Writes the same of 1, plus it visually displays matrices
3. Writes the same of 2, plus the indexes iteration by iteration of the two main functions

---

## Running a single code

### How to compile

> ⚠️ To run the following steps, you must be at least a bit comfortable with a terminal.

-> Clone the repository.
-> You should create the **`build`** folder, if it doesn't exists:
```
mkdir build
```
-> Then choose a code you want to try out and what configurations you want to change, like:
>**CODE:** `Matrix_SymCheck_TranspositionImp1_MA.c` in `src/`
>
>**DEFINES:** `-D SIZE=64 -D DEBUG=1 -D PREFETCH_OFFSET=5`
>
>**FLAGS:** `-fopenmp`
>
>**OUTPUT'S NAME:** `exec_imp1_ma` in `build/`
>

-> Launch this command in the root of the cloned repo:
```
gcc ./src/Matrix_SymCheck_TranspositionImp1_MA.c -o ./build/exec_imp1_ma -D SIZE=64 -D DEBUG=1 -D PREFETCH_OFFSET=5 -fopenmp
```

### How to run
-> After the compilation the above code is run, from the root of the project, with the command:
```
./build/exec_imp1_ma
```
-> The output should look similar to:
```
The selected matrix side size is: 64
Function: checkSymImp1MA
Is matrix Symmetric? -> False
Function: matTransposeImp1MA
Function: matTransposeMA
The two matrices are equal? -> True
Time to check the matrix symmetry (omp_get_wtime) =     0.000072880 sec
Time to transpose the matrix (omp_get_wtime)      =     0.000064671 sec
```

---

## Running full project - cluster

### How the PBS file is structured

### How to run the PBS

---

## Running Python analyzer and grapher
