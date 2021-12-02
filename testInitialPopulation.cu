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

//! \file   testInitialPopulation.cu
//! \brief  file containing the main with the geometric semantic genetic programming algorithm
//! \date   created on 25/01/2020

#include "GsgpCuda.cpp"
#include <cstdio>
#include <ostream>
using namespace std;   

/*!
* \fn       void test(float *individuals, int sizeMaxDepth, int sizePopulation)
* \brief    This file contains the test for the kernel that initializes the individuals on GPU
* \param    float individuals: vector pointers to store the individuals of the initial population.
* \param    int sizeMaxDepth: Variable thar stores maximum depth for individuals.
* \param    int sizePopulation: Size of the population, number of candidate solutions 
* \return   void
* \date     5/12/2020
* \author   Jose Manuel Muñoz Contreras
* \file     testInitialPopulation.cu
*/
void test(float *individuals, int sizeMaxDepth, int sizePopulation){
  bool t;
  for (size_t i = 0; i < sizePopulation; i++){
    for (int j=0; j<sizeMaxDepth; j++){
      if(individuals[i*sizeMaxDepth+j]==-999 || individuals[i*sizeMaxDepth+j]==-0){
        t = false;
      }
      else if(individuals[i*sizeMaxDepth+j]>-999 || individuals[i*sizeMaxDepth+j]>0){
        t = true;
      }
    }
  }
  if(t==true)
  printf("\n Pass Test \n");
  else
  printf("\n Dont Pass Test \n");
}

/*!
* \fn       int main()
* \brief    This file contains the test for the kernel that initializes the individuals on GPU
* \return   int: 0 if the program ends without errors
* \date     5/12/2020
* \author   Jose Manuel Muñoz Contreras
* \file     testInitialPopulation.cu
*/
int main(){
  
  srand(time(NULL)); /*!< initialization of the seed for the generation of random numbers*/

  cudaSetDevice(0); /*!< select a GPU device*/

  printf("\n Starting GsgpCUDA \n\n");

  int individuals = 16; /*!< size of the population, number of candidate solutions */

  const int sizeMaxDepthIndividual = 8; /*!< variable thar stores maximum depth for individuals */

  int sizeMemPopulation = sizeof(float) * individuals * sizeMaxDepthIndividual; /*!< variable that stores size in bytes for initial population*/

  int gridSize,minGridSize,blockSize; /*!< variables that store the execution configuration for a kernel in the GPU*/

  curandState_t* states; /*!< CUDA's random number library uses curandState_t to keep track of the seed value we will store a random state for every thread*/

  cudaMalloc((void**) &states, individuals * sizeof(curandState_t)); /*!< allocate space on the GPU for the random states*/

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, individuals); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/

  gridSize = (individuals + blockSize - 1) / blockSize; /*!< round up according to array size*/

  init<<<gridSize, blockSize>>>(time(0), states); /*!< invoke the GPU to initialize all of the random states*/

  float *dInitialPopulation,*dRandomTrees,*hInitialPopulation, *hRandomTrees;  /*!< This block contains the vectors of pointers to store the population and random trees and space allocation in the GPU*/

  hInitialPopulation = (float *)malloc(sizeMemPopulation);  /*!< allocate space on the CPU for the hInitialPopulation*/

  hRandomTrees = (float *)malloc(sizeMemPopulation); /*!< allocate space on the CPU for the hRandomTrees*/

  checkCudaErrors(cudaMalloc((void **)&dRandomTrees, sizeMemPopulation));  /*!< allocate space on the GPU for the dRandomTrees*/

  checkCudaErrors(cudaMalloc((void **)&dInitialPopulation, sizeMemPopulation)); /*!< allocate space on the GPU for the dInitialPopulation*/

  /*!< invokes the GPU to initialize the initial population*/
  initializePopulation<<< gridSize, blockSize >>>(dInitialPopulation, 1, sizeMaxDepthIndividual, states, 10, 4);

  cudaErrorCheck("initializePopulation");

  /*!<return the initial population of the device to the host*/
  cudaMemcpy(hInitialPopulation, dInitialPopulation, sizeMemPopulation, cudaMemcpyDeviceToHost);

  cudaFree(dInitialPopulation);

  /*!< invokes the GPU to initialize the initial population*/
  initializePopulation<<< gridSize, blockSize >>>(dRandomTrees, 0, sizeMaxDepthIndividual, states, 0, 0);

  cudaErrorCheck("initializePopulation");

  /*!<return the initial population of the device to the host*/
  cudaMemcpy(hRandomTrees, dRandomTrees, sizeMemPopulation, cudaMemcpyDeviceToHost);

  test(hInitialPopulation, sizeMaxDepthIndividual, individuals);

  test(hRandomTrees, sizeMaxDepthIndividual, individuals);

  cudaDeviceReset(); /*!< all device allocations are removed*/

  return 0;
}
