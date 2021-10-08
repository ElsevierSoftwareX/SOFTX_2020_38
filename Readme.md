# Geometric Semantic Genetic Programming into GPU
```
This  is a C/C++/CUDA implementation of a geometric semantic genetic programming algorithm.
```
***
## Parameters:  

Modify the parameters accordingly to adjust to the desired evolutionary conditions

| Name     								| Values   |
| -------- 								| -------- |
|1.  Number of runs						| 1
|2.  Number of generations				| 1024
|3.  Population Size					| 1024
|4.  Maximun tree Depth					| 10
|5.  Number of fitness cases (train)	| 720
|6.  Number of fitness features (train)	| 8
|7.  Number of fitness cases (test)		| 309
|8.  Number of fitness features (test)	| 8
|10. Maximun Random Constant			| 10
|11. Get Row Var From File              | 0
|12. Use Multiple Train Files           | 1
|13. Log Path                           | log/
|14. Data Path Train                    | dataTrain/
|15. Data Path Test                     | dataTest/
|16. Train File                         | train_10107_
|17. Test File                          | test_10107_
|18. Use Test Set                       | 0

## Data Description:  
It is important that the problem data are not separated by ",". Please separate your data by a blank space " ".

## Software code languajes, tools, and services used
```
C/C++/CUDA,CUBLAS
```
## Compilation requirements, operating enviroments & dependencies 
```
Toolkit CUDA v10.1 && v9.2, GCC v7.4.0, CUBLAS v2.0, Linux Headers, unix-like systems, Ubuntu Linux18.04

How to compile.

nvcc -std=c++11 -O0 GsgpCuda.cu -o GsgpCuda.x  -lcublas

How to run.

./GsgpCuda.x

```
***

## How to test best model
```
In the config file update the value of the variable useTestSet = 1

How to run.

./GsgpCuda.x -test_file test.txt -trace_file trace2021-10-08.11:36:25.csv

```
***

## How to run unit tests for the main GsgpCUDA kernels
```
How to compile for kernel unit tests that initialize the population.

 nvcc -std=c++11 -O0 testInitialPopulation.cu -o testInitialPopulation.x

How to run.

./testInitialPopulation.x

How to compile for kernel unit tests that calculate the semantics.

 nvcc -std=c++11 -O0 testSemantic.cu -o testSemantic.x 

How to run.

./testSemantic.x 

How to compile for kernel unit tests that executes the semantic geometric mutation operator.

 nvcc -std=c++11 -O0 gsmTest.cu -o gsmTest.x

How to run.

./gsmTest.x
```
## Documentation:
```
The documentation of the library is a Doxygen documentation. The implementation has been done in order to use the library after a very quick reading of the documentation.
```