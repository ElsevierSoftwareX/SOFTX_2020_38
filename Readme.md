# Geometric Semantic Genetic Programming into GPU
```
This  is a C/C++/CUDA implementation of a geometric semantic genetic programming algorithm.
```
***
## Parameters:  

Modify the parameters accordingly to adjust to the desired evolutionary conditions

| Name     								| Values   |
| -------- 								| -------- |
|1.  Number of generations				| 1024
|2.  Population Size					| 1024
|3.  Maximun tree Depth					| 10
|4. Maximun Random Constant			| 10
|5. Log Path                           | log/


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

To run gsgpCuda it is necessary to add a name for the output file generation, as shown in the example.

./GsgpCuda.x -train_data train.txt -test_data test.txt -output_model best_model

train.txt: This file must contain the training data.
test.txt:  This file must contain the test data.
best_model: This file contains the information needed to test the model generated by GsgpCuda.
    best_model_initialPopulation.csv: This file will store the individuals of the initial population.
    best_model_randomTrees.csv: This file will store the individuals of the auxiliary population.
    best_model_fitnessTrain.csv: This file will store the error of the best individual in each generation with training data.
    best_model_fitnessTest.csv: This file will store the error of the best individual in each generation with test data.
    best_model_processing_time.csv: This file stores the processing times in seconds of the various modules of the algorithm. 


```
***

## How to test best model
```
To test the model generated by GsgpCuda it is necessary to provide the name of the model by command line, 
the second parameter indicates the name of the data file, the third parameter indicates the name of the file to save the output values generated by the model.

./GsgpCuda.x -model XXXXXX -input_data YYYYYYY -prediction_output ZZZZZZ

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