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

//! \file   testSemantic.cu
//! \brief  This file contains the test for parallel interpreter that is used to decode the GP individuals.
//! \date   created on 05/12/2020

#include "GsgpCuda.cpp"
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <stack>
using namespace std;   

/*!
* \fn       void individualsTest(float *caseTest, int caseOption, int depth, int numIndi, int nvar)
* \brief    This function initializes the individuals in the population depending on the case, they can be addition, subtraction, multiplication and division.
* \param    float *caseTest: vector pointers to store the individuals of the initial population
* \param    int caseOption: This variable determines what type of individuals will be generated
* \param    int depth: Variable thar stores maximum depth for individuals
* \param    int *numIndi: This variable contains the number of individuals that exist in the initial population.
* \param    int *nrow: This variable contains the number of fitness cases.
* \param    int *nvar: This variable contains the number of variable in problem.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     testSemantic.cu
*/
void individualsTest(float *caseTest, int caseOption, int depth, int numIndi, int nvar){
    srand(time(NULL));
    int ale;
    if (caseOption==1) {
        printf("Print case one\n");
        for (int j=0; j<numIndi; j++) {
            for (int i=0; i<depth; i++) {
                ale = rand()%(3);
                if (ale==1) {
                    caseTest[j*depth+i]=-1;
                }else if(ale==2){
                    caseTest[j*depth+i]=10;
                }else {
                    caseTest[j*depth+i]=-1000;
                }    
            }   
        }
    }else if (caseOption==2) {
        printf("Print case two \n");
        for (int j=0; j<numIndi; j++) {
            for (int i=0; i<depth; i++) {
                ale = rand()%(3);
                if (ale==1) {
                    caseTest[j*depth+i]=-2;
                }else if(ale==2){
                    caseTest[j*depth+i]=10;
                }else {
                    caseTest[j*depth+i]=-1000;
                }    
            }   
        }
    }else if (caseOption==3) {
        printf("Print case three \n");
        for (int j=0; j<numIndi; j++) {
            for (int i=0; i<depth; i++) {
                ale = rand()%(3);
                if (ale==1) {
                    caseTest[j*depth+i]=-3;
                }else if(ale==2){
                    caseTest[j*depth+i]=1;
                }else {
                    caseTest[j*depth+i]=-1000;
                }    
            }   
        }
    }else if (caseOption==4) {
        printf("Print case four%d \n",caseOption);
        for (int j=0; j<numIndi; j++) {
            for (int i=0; i<depth; i++) {
                ale = rand()%(3);
                if (ale==1) {
                    caseTest[j*depth+i]=-4;
                }else if(ale==2){
                    caseTest[j*depth+i]=1;
                }else {
                    caseTest[j*depth+i]=-1000;
                }    
            }   
        }
    }
}

/*!
* \fn       void resetIndi(float *caseTest, int numIndi, int depth)
* \brief    This function initializes all elements of the individual to 0.
* \param    float *caseTest: This vector of pointers contains the semantics of the initial population or random trees as the case may be.
* \param    int *numIndi: This variable contains the number of individuals that exist in the initial population.
* \param    int *depth: Variable thar stores maximum depth for individuals.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     testSemantic.cu
*/
void resetIndi(float *caseTest, int numIndi, int depth){
    for (int i=0; i<numIndi; i++) {
        for (int j=0; j<depth; j++) {
            caseTest[i*depth+j]=0;
        }
    }
}

/*!
* \fn       void fitnessCases(float *fitCases, int nrow)
* \brief    This function only generates an aptitude case to test individuals.
* \param    float *fitCases:This vector of pointers contains the instances to train or fitnes cases.
* \param    int *nrow: This variable contains the number of fitness cases.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     testSemantic.cu
*/
void fitnessCases(float *fitCases, int nrow){
    for (int i=0; i<nrow; i++) {
        fitCases[i]=2;
    }
}

/*!
* \fn       void imprimeIndividuals(float *caseTest, int numIndi, int depth)
* \brief    This function initializes with random numbers between 0 and 1 the semantics for initial population and random trees.
* \param    float *caseTest:This vector of pointers contains the indivuduals of the initial population.
* \param    int *numIndi: This variable contains the number of individuals that exist in the initial population.
* \param    int *depth: Variable thar stores maximum depth for individuals.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     testSemantic.cu
*/
void imprimeIndividuals(float *caseTest, int numIndi, int depth){
    for (int i=0; i<numIndi; i++) {
        for (int j=0; j<depth; j++) {
            printf(" %f,", caseTest[i*depth+j]);
        }
        printf("\n");
    }
}

/*!
* \fn       void imprimeSemantic(float *caseTest, int numIndi, int depth)
* \brief    This function print the semantic of the initial population.
* \param    float *devSemanticTrainCases:This vector of pointers contains the semantic of the initial population.
* \param    int *numIndi: This variable contains the number of individuals that exist in the initial population.
* \param    int *depth: Variable thar stores maximum depth for individuals.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     testSemantic.cu
*/
void imprimeSemantic(float *devSemanticTrainCases, int numIndi, int depth){
    for (int i=0; i<numIndi; i++) {
        for (int j=0; j<depth; j++) {
            printf(" %f,", devSemanticTrainCases[i*depth+j]);
        }
        printf("\n");
    }
}

/*!
* \fn       void clear(stack <float> as)
* \brief    remove all elements from the stack so that in the next evaluations there are no previous values of other individuals
* \param    stack as: auxiliary pointer that stores the values ​​resulting from the evaluation of each individual
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/
void clean(stack <float> as){
    int a = as.size();
    while (!as.empty()){
        as.pop();
    }
}

/*!
* \fn       void intreSemanticCPU(float *initiPop, float *OutSemantic, float *data, int nrow, int depth, int numIndi)
* \brief    This function interprets the generated individuals in CPU to obtain their semantics.
* \param    float *initiPop: This vector pointers to store the individuals of the initial population.
* \param    flot  *OutSemantic: vector pointers to store the semantics of each individual in the population.
* \param    float *data: This pointer vector containing training or test data.
* \param    int *nrow: This variable contains the number of fitness cases.
* \param    int depth: This variable thar stores maximum depth for individuals
* \param    int numIndi: This variable contains the number of individuals that exist in the initial population.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     testSemantic.cu
*/
void intreSemanticCPU(float *initiPop, float *OutSemantic, float *data, int nrow, int depth, int numIndi){
    float tmp,tmp2,out;
    stack <float> a;
    for (int l=0; l<nrow; l++) {
        out=0;
        clean(a);
        a = stack<float>();
        clear(a);
        if (a.empty()) {
            for (int i=0; i<numIndi; i++) {
                for (int j=0; j<depth; j++) {
                    if (initiPop[i*depth+j]>0) {
                        a.push(initiPop[i*depth+j]);
                    }else if (initiPop[i*depth+j] <= -1000) {
                        a.push(data[l]);
                    } else if (initiPop[i*depth+j]== -1) {
                        if (!a.empty()) {
                            tmp = a.top();
                            a.pop();
                        } if (!a.empty()) {
                            tmp2 = a.top();
                            a.pop();
                            a.push(tmp+tmp2);
                            out=tmp+tmp2;  
                        } else {
                            a.push(tmp);
                        }          
                    }  else if (initiPop[i*depth+j]== -2) {
                        if (!a.empty()) {
                            tmp = a.top();
                            a.pop();
                        } if (!a.empty()) {
                            tmp2 = a.top();
                            a.pop();
                            out=tmp-tmp2;  
                            a.push(out);
                        } else {
                            a.push(tmp);
                        }          
                    } else if (initiPop[i*depth+j]== -3) {
                        if (!a.empty()) {
                            tmp = a.top();
                            a.pop();
                        } if (!a.empty()) {
                            tmp2 = a.top();
                            a.pop();
                            out=tmp*tmp2;
                            a.push(out);
                              
                        } else {
                            a.push(tmp);
                        }          
                    } else if (initiPop[i*depth+j]== -4) {
                        if (!a.empty()) {
                            tmp = a.top();
                            a.pop();
                        } if (!a.empty()) {
                            tmp2 = a.top();
                            a.pop();
                            out=tmp2 / sqrt(1+tmp*tmp);
                            a.push(tmp2 / sqrt(1+tmp*tmp));
                        }else {
                            a.push(tmp);
                        }
                    }
                }
                OutSemantic[i*nrow+l] = out;
            }
        
        }
    }
    
}

/*!
* \fn       void testSemantic(float *hostSemanticTrainCases, float *cpuSemantic, int num, int dept)
* \brief    This function determines and compares if the semantics on GPU and CPU are identical.
* \param    float *hostSemanticTrainCases:This vector of pointers contains the semantics of the initial population or random trees as the case may be.
* \param    float *cpuSemantic:This vector of pointers contains the semantics of the initial population or random trees as the case may be.
* \param    int num: This variable contains the number of individuals that exist in the initial population.
* \param    int depth: This variable thar stores maximum depth for individuals
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     testSemantic.cu
*/
void testSemantic(float *hostSemanticTrainCases, float *cpuSemantic, int num, int dept){
    bool test;
    for (int i=0; i<num; i++) {
        for (int j=0; j<dept; j++) {
            if(hostSemanticTrainCases[i*dept+j] == cpuSemantic[i*dept+j]){
                test = true;
                printf("Pass Test \n");
            }else {
                test=false;
                printf("Dont Pass Test \n");
            }
        }
    }

}


/*!
* \fn       int main(int argc, const char **argv)
* \brief    main method that runs the test algorithm computeSemantics
* \return   int: 0 if the program ends without errors
* \date     25/01/2020
* \author   Jose Manuel Muñoz Contreras
* \file     testSemantic.cu
*/
int main(){

    srand(time(NULL)); /*!< initialization of the seed for the generation of random numbers*/

    cudaSetDevice(0); /*!< select a GPU device*/

    printf("\n Starting GsgpCUDA \n\n");

    int caseoption=4;
    
    int dept=8; /*!< variable that stores maximum depth for individuals */
    
    int num=1; /*!< size of the population, number of candidate solutions */ 
    
    int nrow =8; /*!< variable containing the number of rows (instances) of the training dataset */

    int size = sizeof(float) * num * dept; /*!< variable that stores size in bytes for initial population*/
    
    int sizeMemIndividuals = sizeof(float) * num; /*!< variable that stores size in bytes of the number of individuals in the initial population*/

    int sizeMemPopulation = sizeof(float) * num*dept; /*!< variable that stores size in bytes for initial population*/

    int sizeSemantic = sizeof(float)*num*nrow; /*!< variable that stores the size in bytes of semantics for the entire population with training data*/

    int sizeT = sizeof(float)*nrow;  /*!< variable that stores the size in bytes of fitness cases*/

    int gridSize,minGridSize,blockSize; /*!< variables that store the execution configuration for a kernel in the GPU*/
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeSemantics, 0, num); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
    
    gridSize = (num + blockSize - 1) / blockSize; /*!< round up according to array size*/

    printf("Grid and block %d %d \n", gridSize, blockSize );

    float *caseTest, *devCaseTest, *devSemanticTrainCases, *hostSemanticTrainCases, *trainingTest, *devtrainingTest, *cpuSemantics; /*Pointers of vectos to use*/

    hostSemanticTrainCases = (float *)malloc(sizeSemantic); /*!< allocate space on the CPU for the hostSemanticTrainCases*/

    cpuSemantics = (float *)malloc(sizeSemantic); /*!< allocate space on the CPU for the cpuSemantics*/

    cudaMalloc((void **)&devSemanticTrainCases,sizeSemantic); /*!< allocate space on the GPU for the devSemanticTrainCases*/
     
    caseTest = (float *)malloc(size); /*!< allocate space on the GPU for the caseTest*/
 
    cudaMalloc((void**)&devCaseTest,sizeof(float)*size); /*!< allocate space on the GPU for the devCaseTest*/

    trainingTest = (float *)malloc(sizeT); /*!< allocate space on the CPU for the trainingTest*/

    cudaMalloc((void **)&devtrainingTest,sizeof(float)*nrow);   /*!< allocate space on the CPU for the devtrainingTest*/

    float *uStackInd; /*!< auxiliary pointer vectors for the interpreter and calculate the semantics for the populations and assignment in the GPU*/
    
    int   *uPushGenes;
    
    checkCudaErrors(cudaMalloc((void**)&uPushGenes, sizeMemIndividuals));  /*!< allocate space on the GPU for the uPushGenes*/
    
    checkCudaErrors(cudaMalloc((void**)&uStackInd, sizeMemPopulation));  /*!< allocate space on the CPU for the uStackInd*/

    fitnessCases(trainingTest, nrow); /*!<This function only generates an aptitude case to test individuals.>*/
    
    cudaMemcpy(devtrainingTest,trainingTest,sizeof(float)*nrow,cudaMemcpyHostToDevice);  /*!< This instruction makes a copy of a vector of pointers from CPU to GPU. */

    for (int i=1; i<=caseoption; i++) {
        individualsTest(caseTest, i, dept, num, 10); /*!<This function initializes the individuals in the population depending on the case, they can be addition, subtraction, multiplication and division.>*/
        
        intreSemanticCPU(caseTest, cpuSemantics, trainingTest, nrow, dept, num); /*!<This function that decodes each individual and evaluates it over all fitness cases,>*/
        
        cudaMemcpy(devCaseTest,caseTest,sizeof(float)*size,cudaMemcpyHostToDevice);   /*!< This instruction makes a copy of a vector of pointers from CPU to GPU. */

        computeSemantics<<< gridSize, blockSize >>>(devCaseTest, devSemanticTrainCases, dept, devtrainingTest, nrow, 1, uPushGenes, uStackInd); /*!< The ComputeSemantics kernel is an interpreter, that decodes each individual and evaluates it over all fitness cases,>*/
        
        cudaErrorCheck("computeSemantics");
        
        cudaMemcpy(hostSemanticTrainCases,devSemanticTrainCases,sizeSemantic,cudaMemcpyDeviceToHost);   /*!< This instruction makes a copy of a vector of pointers from GPU to CPU. */
        
        resetIndi(caseTest, num, dept); /*!<This function initializes all elements of the individual to 0.>*/
        
        testSemantic(hostSemanticTrainCases, cpuSemantics, num, dept); /*!<This function determines and compares if the semantics on GPU and CPU are identical.>*/
    }
    
    free(hostSemanticTrainCases);
    free(caseTest);
    free(trainingTest);
    free(cpuSemantics);
    cudaFree(devCaseTest);
    cudaFree(devSemanticTrainCases);
    cudaFree(uPushGenes);
    cudaFree(uStackInd);
    cudaDeviceReset(); /*!< all device allocations are removed*/
    
    return 0;
}
