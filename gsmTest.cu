/*<one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2020 José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

     This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//! \file   omsTest.cu
//! \brief  file containing the main test with the geometric semantic operator algorithm
//! \date   created on 25/01/2020

#include "GsgpCuda.cpp"
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <stack>
/// Macro used to generate a random number
#define frand() ((double) rand() / (RAND_MAX))
using namespace std;   

/*!
* \fn       int main(int argc, const char **argv)
* \brief    main method that executes the test for the semantic geometric mutation operator
* \param    int argc: sizeIndividualsber of parameters of the program
* \param    const char **argv: array of strings that contains the parameters of the program
* \return   int: 0 if the program ends without errors
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cu
*/

/*!
* \fn       void exampleSemantics(float *semantic, int sizeIndividuals, int nrow)
* \brief    This function initializes with random numbers between 0 and 1 the semantics for initial population and random trees.
* \param    float *semantic:This vector of pointers contains the semantics of the initial population or random trees as the case may be.
* \param    int *sizeIndividuals: This variable contains the number of individuals that exist in the initial population.
* \param    int *nrow: This variable contains the number of fitness cases.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     omsTest.cu
*/
void exampleSemantics(float *semantic, int sizeIndividuals, int nrow){
    for (int i=0; i<sizeIndividuals; i++) {
        for (int j=0; j<nrow; j++) {
            semantic[i*nrow+j] = frand();            
        }
    }
}

/*!
* \fn       void printSemantics(float *semantic, int sizeIndividualsIndi, int nrow)
* \brief    This function only prints the semantic values that exist in the pointer vectors it receives.
* \param    float *semantic:This vector of pointers contains the semantics of the initial population or random trees as the case may be.
* \param    int *sizeIndividuals: This variable contains the number of individuals that exist in the initial population.
* \param    int *nrow: This variable contains the number of fitness cases.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     omsTest.cu
*/
void printSemantics(float *semantic, int sizeIndividualsIndi, int nrow){
    for (int i=0; i<sizeIndividualsIndi; i++) {
        for (int j=0; j<nrow; j++) {
            printf(" %f,", semantic[i*nrow+j]);
        }
        printf("\n");
    }
}

/*!
* \fn       void omsCPU(float *initialPopSemantic, float *randomTreesSemantic, float *newOffspringSemantic, int sizePopulation ,int nrow, double ms, float *index)
* \brief    The GSM operator is basically a vector addition operation, that can be performed independently for each semantic element STi,j.
            However, it is necessary to select the semantics of two random trees R u and R v , and a random mutation step ms.
* \param    float *initialPopSemantic: This vector of pointers contains the semantics of the initial population
* \param    float *randomTreesSemantic: This vector of pointers contains the semantics of the random trees
* \param    float *newOffspringSemantic: This vector of pointers will store the semantics of the new offspring
* \param    int sizePopulation: This variable contains the number of individuals that the population has
* \param    int nrow: Variable containing the number of rows (instances) of the training dataset
* \param    double ms: This variable stores a random value that causes a disturbance in the sematics of an individual
* \param    int generation: Number of generation
* \param    float *indexRandomTrees: This pointer stores the indexes randomly for mutation
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     omsTest.cu
*/
void omsCPU(float *initialPopSemantic, float *randomTreesSemantic, float *newOffspringSemantic, int sizePopulation ,int nrow, double ms, float *index){
    int seed;
    int firstTree, secondTree;
    for (int i=0; i<sizePopulation; i++) {
        for (int j=0; j<nrow; j++) {
            seed = i+j/nrow;
            firstTree = index[seed];
            secondTree = index[sizePopulation + seed];
            float sigmoid_1=1.0/(1+std::exp(-(randomTreesSemantic[firstTree*nrow+j]))); 
            float sigmoid_2=1.0/(1+std::exp(-(randomTreesSemantic[secondTree*nrow+j])));
            newOffspringSemantic[i*nrow+j] = initialPopSemantic[i*nrow+j] + (ms *(sigmoid_1-sigmoid_2));
        } 
    }
}

/*!
* \fn       void testSemantic(float *cpuSemantic, float *gpuSemantic, int sizeIndividuals, int nrow)
* \brief    This function compares the semantics obtained in the GPU against a semantics obtained in the CPU to verify that the operations within the kernel executed in the GPU are correct.
* \param    float *cpuSemantic:This vector of pointers contains the semantics of the new individuals resulting from the initial population and random trees, calculated in CPU.
* \param    float *gpuSemantic:This vector of pointers contains the semantics of the new individuals resulting from the initial population and random trees, calculated in GPU.
* \param    int *sizeIndividuals: This variable contains the number of individuals that exist in the initial population.
* \param    int *nrow: This variable contains the number of fitness cases.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     omsTest.cu
*/
bool testSemantic(float *cpuSemantic, float *gpuSemantic, int sizeIndividuals, int nrow){
    bool test;
    for (int i=0; i<sizeIndividuals; i++) {
        for (int j=0; j<nrow; j++) {
            if(cpuSemantic[i*nrow+j] == gpuSemantic[i*nrow+j]){
                test = true;
                //printf("Pass Test \n");
            }else {
                test=false;
                //printf("Dont Pass Test \n");
            }
        }
    }
    return test;
}

int main(){

    srand(time(NULL)); /*!< initialization of the seed for the generation of random sizeIndividualsbers*/

    cudaSetDevice(0); /*!< select a GPU device*/

    printf("\n Starting GsgpCUDA \n\n");

    int sizeIndividuals=1; /*!< size of the population, number of candidate solutions */ 
    
    int nrow =8; /*!< variable containing the number of rows (instances) of the training dataset */

    int sizeSemantic = sizeof(float)*sizeIndividuals*nrow; /*!< variable that stores the size in bytes of semantics for the entire population with training data*/

    int sizeElementsSemanticTrain=sizeIndividuals*nrow; /*!< variable that stores training data elements*/

    int twoSizeMemPopulation = sizeof(float) * (sizeIndividuals*2); /*!< variable that stores twice the size in bytes of an initial population to store random numbers*/

    long int twoSizePopulation = (sizeIndividuals*2); /*!< variable storing twice the initial population of individuals to generate random positions*/

    size_t structMemMutation = (sizeof(entry_)*sizeIndividuals); /*!< variable that stores the size in bytes of the structure to store the mutation record*/

    entry  *dStructMutation; /*!< This block contains the vectors of pointers to store the structure to keep track of mutation and survival and space allocation in the GPU*/
    checkCudaErrors(cudaMallocManaged(&dStructMutation,structMemMutation)); 

    entry  *dStructMutationy; /*!< This block contains the vectors of pointers to store the structure to keep track of mutation and survival and space allocation in the GPU*/
    checkCudaErrors(cudaMallocManaged(&dStructMutationy,structMemMutation)); 

    int gridSize,minGridSize,blockSize; /*!< variables that store the execution configuration for a kernel in the GPU*/
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeSemantics, 0, sizeIndividuals); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
    
    gridSize = (sizeIndividuals + blockSize - 1) / blockSize; /*!< round up according to array size*/

    printf("Grid and block %d %d \n", gridSize, blockSize );

    float *cpuSemanticInitialPopulation, *gpuSemanticInitialPopulation, *cpuRandomTrees, 
    *gpuRandomTrees, *cpuNewSemanticsOffsprings, *gpuNewSemanticsOffsprings, *hostNewSemanticsOffsprings;

    cpuSemanticInitialPopulation = (float *)malloc(sizeSemantic); /*!< allocate space on the CPU for the cpuSemanticInitialPopulation*/

    cpuRandomTrees = (float *)malloc(sizeSemantic); /*!< allocate space on the CPU for the cpuRandomTrees*/

    cpuNewSemanticsOffsprings = (float *)malloc(sizeSemantic);  /*!< allocate space on the CPU for the cpuNewSemanticsOffsprings*/

    hostNewSemanticsOffsprings = (float *)malloc(sizeSemantic); /*!< allocate space on the GPU for the hostNewSemanticsOffsprings*/

    exampleSemantics(cpuSemanticInitialPopulation, sizeIndividuals, nrow);  /*!< Invoke the function to initialize the random semantics for the initial population.*/

    printf("Print initial semantic \n");
    
    printSemantics( cpuSemanticInitialPopulation, sizeIndividuals, nrow); /*!< Invoke the function print the random semantics for the initial population.*/

    exampleSemantics(cpuRandomTrees, sizeIndividuals,  nrow); /*!< Invoke the function to initialize the random values semantics for the random trees.*/
    
    printf("Print initial semantic\n");
    
    printSemantics(cpuRandomTrees, sizeIndividuals, nrow); /*!< Invoke the function print the random semantics for the the random trees.*/

    cudaMalloc((void **)&gpuSemanticInitialPopulation,sizeSemantic);  /*!< allocate space on the GPU for the gpuSemanticInitialPopulation*/
    
    cudaMalloc((void**)&gpuRandomTrees,sizeSemantic); /*!< allocate space on the GPU for the gpuRandomTrees*/

    cudaMalloc((void**)&gpuNewSemanticsOffsprings,sizeSemantic); /*!< allocate space on the GPU for the gpuNewSemanticsOffsprings*/

    cudaMemcpy(gpuSemanticInitialPopulation,cpuSemanticInitialPopulation,sizeSemantic,cudaMemcpyHostToDevice); /*!< This instruction makes a copy of a vector of pointers from CPU to GPU. */

    cudaMemcpy(gpuRandomTrees,cpuRandomTrees,sizeSemantic,cudaMemcpyHostToDevice); /*!< This instruction makes a copy of a vector of pointers from CPU to GPU. */

    curandState_t* State; /*!< CUDA's random number library uses curandState_t to keep track of the seed value we will store a random state for every thread*/
    
    cudaMalloc((void**) &State, (twoSizePopulation) * sizeof(curandState_t)); /*!< allocate space on the GPU for the random states*/
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, twoSizePopulation); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
    
    gridSize = (twoSizePopulation + blockSize - 1) / blockSize; /*!< round up according to array size*/
    
    init<<<gridSize, blockSize>>>(time(NULL), State); /*!< invoke the GPU to initialize all of the random states*/

    float *indexRandomTrees; /*!< vector pointers to save random positions of random trees and allocation in GPU*/
    checkCudaErrors(cudaMallocManaged(&indexRandomTrees,twoSizeMemPopulation));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initializeIndexRandomTrees, 0, twoSizePopulation); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
    gridSize = (twoSizePopulation + blockSize - 1) / blockSize;  /*!< round up according to array size*/
            
    initializeIndexRandomTrees<<<gridSize,blockSize >>>( sizeIndividuals, indexRandomTrees, State ); 
    
    cudaErrorCheck("initializeIndexRandomTrees");

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, geometricSemanticMutation, 0, sizeElementsSemanticTrain);  /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
    
    gridSize = (sizeElementsSemanticTrain + blockSize - 1) / blockSize; /*!< round up according to array size*/
    
    printf("Grid and block Mutacion %d %d \n", gridSize, blockSize );
    /*!< geometric semantic mutation with semantic train*/
    geometricSemanticMutation<<< gridSize, blockSize >>>(gpuSemanticInitialPopulation, gpuRandomTrees,gpuNewSemanticsOffsprings,
        sizeIndividuals, nrow, sizeElementsSemanticTrain, 1, indexRandomTrees, dStructMutation, dStructMutationy);
    
        cudaErrorCheck("geometricSemanticMutation");

    cudaMemcpy(hostNewSemanticsOffsprings,gpuNewSemanticsOffsprings,sizeSemantic,cudaMemcpyDeviceToHost);  /*!< This instruction makes a copy of a vector of pointers from CPU to GPU. */
    
    printf("Print initial semantic of GPU \n");
    
    printSemantics(hostNewSemanticsOffsprings, sizeIndividuals, nrow);
    
    float m= dStructMutation[0].mutStep;

    omsCPU(cpuSemanticInitialPopulation, cpuRandomTrees, cpuNewSemanticsOffsprings, sizeIndividuals, nrow, m, indexRandomTrees);
    
    printf(" Print initial semantic of CPU \n");    
    
    printSemantics(cpuNewSemanticsOffsprings, sizeIndividuals,  nrow);

    if(testSemantic(cpuNewSemanticsOffsprings,hostNewSemanticsOffsprings,sizeIndividuals,nrow)){
        printf(" Pass test for the semantic mutation operator \n");
    }else{
        printf(" Don't pass test for the semantic mutation operator \n");
    }
    
    free(cpuSemanticInitialPopulation);
    free(cpuRandomTrees);
    free(cpuNewSemanticsOffsprings);
    free(hostNewSemanticsOffsprings);
    cudaFree(gpuSemanticInitialPopulation);
    cudaFree(gpuRandomTrees);
    cudaFree(gpuNewSemanticsOffsprings);
    cudaDeviceReset(); /*!< all device allocations are removed*/
    
    return 0;
}
