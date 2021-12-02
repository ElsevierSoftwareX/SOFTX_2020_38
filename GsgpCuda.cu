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

//! \file   GsgpCuda.cu
//! \brief  file containing the main with the geometric semantic genetic programming algorithm
//! \author Jose Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
//! \date   created on 25/01/2020
#include "GsgpCuda.cpp"

/*!
* \fn       int main(int argc, const char **argv)
* \brief    main method that runs the GSGP algorithm and test the best model generate by GSGP-CUDA
* \param    int argc: number of parameters of the program
* \param    const char **argv: array of strings that contains the parameters of the program
* \return   int: 0 if the program ends without errors
* \date     25/01/2020
* \author   Jose Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cu
*/
int main(int argc, char **argv){

    printf("\n Starting GsgpCuda \n\n");
    
    srand(time(NULL)); /*!< Initialization of the seed for the generation of random numbers*/

    readConfigFile(&config); /*!< reading the parameters of the algorithm */

    cudaSetDevice(0); /*!< Select a GPU device*/
    
    char trainFile[50]="";    /*!< Name of the train file*/
    char testFile[50]="";     /*!< Name of the test file*/
    char output_model[50]=""; /*!< Name of output files*/
    char pathTrace[50]="";    /*!< Name of the file trace of best model*/
    char path_test[50]="";    /*!< Name of the file with unsseen test instances*/
    char pathOutFile[50]="";  /*!< Name of the file to output values*/
    for (int i=1; i<argc-1; i++){
        if(strncmp(argv[i],"-train_file",10) == 0) {
            strcat(trainFile,argv[++i]);
        }else if (strncmp(argv[i],"-test_file",10) == 0) {
            strcat(testFile,argv[++i]);
        }else if (strncmp(argv[i],"-output_model",10)==0) {
            strcat(output_model,argv[++i]);
        }else if (strncmp(argv[i],"-model",10)==0) {
            strcat(pathTrace,argv[++i]);
        }else if (strncmp(argv[i],"-input_data",10)==0) {
            strcat(path_test,argv[++i]);
        }else if (strncmp(argv[i],"-prediction_output",10)==0) {
            strcat(pathOutFile,argv[++i]);
        }     
    }

    std::string trainFile_s(trainFile);
 
    std::string testFile_s(testFile);
 
    std::string outputNameFiles(output_model); ///*!< Name of file for save the output files*/
 
    std::string la (pathTrace);
 
    std::string logPath (config.logPath); /* Path of directory for data files and log files generated in execution */
 
    std::string namePopulation = "_initialPopulation"; /*!< Name of file for save the initial population  */

    namePopulation = outputNameFiles + namePopulation;

    std::string nameRandomTrees = "_randomTrees"; /*!< name of file for save the random trees */
    
    nameRandomTrees = outputNameFiles + nameRandomTrees; 
        
    if (!trainFile_s.empty() && testFile_s.empty()){
        
        countInputFile(trainFile, nrow, nvar); ///Counting the number of rows and variables of the train file
        
        nvar--; 

        individualLength = config.maxIndividualLength; /*!< Variable that stores maximum depth for individuals */

        sizeMemIndividuals = sizeof(float) * config.populationSize; /*!< Variable that stores size in bytes of the number of individuals in the initial population*/

        twoSizeMemPopulation = sizeof(float) * (config.populationSize*2); /*!< Variable that stores twice the size in bytes of an initial population to store random numbers*/

        sizeMemPopulation = sizeof(float) * config.populationSize * individualLength; /*!< Variable that stores size in bytes for initial population*/
        
        twoSizePopulation = (config.populationSize*2); /*!< Variable storing twice the initial population of individuals to generate random positions*/

        sizeMemSemanticTrain = sizeof(float)*(config.populationSize*nrow); /*!< Variable that stores the size in bytes of semantics for the entire population with training data*/

        sizeMemDataTrain = sizeof(float)*(nrow*nvar); /*!< Variable that stores the size in bytes the size of the training data*/

        sizeElementsSemanticTrain = (config.populationSize*nrow); /*!< Variable that stores training data elements*/

        vectorTracesMem = (sizeof(entry_)*config.numberGenerations*config.populationSize); /*!< Variable that stores the size in bytes of the structure to store the survival record*/

        std::string logPath (config.logPath); /* Path of directory for data files and log files generated in execution */

        std::string namePopulation = "_initialPopulation"; /*!< Name of file for save the initial population  */

        namePopulation = outputNameFiles + namePopulation;

        std::string nameRandomTrees = "_randomTrees"; /*!< name of file for save the random trees */
        
        nameRandomTrees = outputNameFiles + nameRandomTrees;

        /* Check if log and data diectories exists */
        checkDirectoryPath(logPath);
        
        float executionTime = 0, initialitionTimePopulation = 0, timeComputeSemantics = 0, generationTime = 0; /*!< Variables that store the time in milliseconds between the events mark1 and mark2.*/

        std::string timeExecution1 = "_processing_time"; /*!< Variable name structure responsible for indicating the run*/
        std::string timeExecution2 = ".csv"; /*!< Variable name structure responsible for indicating the file extension*/
        timeExecution1 = logPath + outputNameFiles + timeExecution1 + timeExecution2; /*!< Variable that stores file name matching*/
        std::ofstream times(timeExecution1,ios::out); /*!< pointer to the timeExecution1 file that contains the time consumed by the different algorithm modules*/
 
        cudaEvent_t startRun, stopRun;  /*!< Variable used to create a start mark and a stop mark to create events*/
        cudaEventCreate(&startRun);     /*!< function that initializes the start event*/
        cudaEventCreate(&stopRun);      /*!< function that initializes the stop event*/

        curandState_t* states; /*!< CUDA's random number library uses curandState_t to keep track of the seed value we will store a random state for every thread*/
        cudaMalloc((void**) &states, config.populationSize * sizeof(curandState_t)); /*!< allocate space on the GPU for the random states*/
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, config.populationSize); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/

        gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/
        init<<<gridSize, blockSize>>>(time(0), states); /*!< invoke the GPU to initialize all of the random states*/

        cudaEventRecord(startRun);     
        std::string fitnessTrain  = "_fitnestrain"; /**/
        std::string fitnessTrain2 = ".csv"; /**/
        fitnessTrain = logPath + outputNameFiles + fitnessTrain + fitnessTrain2; /**/
        std::ofstream fitTraining(fitnessTrain,ios::out); /*!< pointer to the file fitnesstrain.csv containing the training fitness of the best individual at each generation*/

        cublasHandle_t handle; /*!< the handle to the cuBLAS library context*/
        cublasCreate(&handle); /*!< initialized using the function and is explicitly passed to every subsequent library function call*/
 
        hInitialPopulation = (float *)malloc(sizeMemPopulation); 
        hRandomTrees = (float *)malloc(sizeMemPopulation); 
        checkCudaErrors(cudaMalloc((void **)&dRandomTrees, sizeMemPopulation)); 
        checkCudaErrors(cudaMalloc((void **)&dInitialPopulation, sizeMemPopulation));
        checkCudaErrors(cudaMallocManaged(&vectorTraces,vectorTracesMem));
        checkCudaErrors(cudaMallocManaged(&uDataTrain, sizeMemDataTrain));     
        checkCudaErrors(cudaMallocManaged(&uDataTrainTarget, sizeof(float)*nrow));            
        checkCudaErrors(cudaMallocManaged(&uFit, sizeMemIndividuals));  
        checkCudaErrors(cudaMallocManaged(&uSemanticTrainCases,sizeMemSemanticTrain));       
        checkCudaErrors(cudaMallocManaged(&uSemanticRandomTrees,sizeMemSemanticTrain));      
        checkCudaErrors(cudaMalloc((void**)&uPushGenes, sizeMemIndividuals));
        checkCudaErrors(cudaMalloc((void**)&uStackInd, sizeMemPopulation));            
       
        readInpuDataTrain(trainFile, uDataTrain, uDataTrainTarget, nrow, nvar); /// load set data train **/
        
        gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/            
        
        cudaEvent_t startInitialPop, stopInitialPop; /*!< this section declares and initializes the Variables for the events and captures the time elapsed in the initialization of the initial population in the GPU*/
        cudaEventCreate(&startInitialPop);
        cudaEventCreate(&stopInitialPop);
        cudaEventRecord(startInitialPop);

        ///invokes the GPU to initialize the initial population
        initializePopulation<<< gridSize, blockSize >>>(dInitialPopulation, nvar, individualLength, states, config.maxRandomConstant, 4, config.functionRatio, config.variableRatio);
        cudaErrorCheck("initializePopulation");

        cudaEventRecord(stopInitialPop);
        cudaEventSynchronize(stopInitialPop);
        cudaEventElapsedTime(&initialitionTimePopulation, startInitialPop, stopInitialPop);
        cudaEventDestroy(startInitialPop);
        cudaEventDestroy(stopInitialPop);    
        ///*!<return the initial population of the device to the host*/
        cudaMemcpy(hInitialPopulation, dInitialPopulation, sizeMemPopulation, cudaMemcpyDeviceToHost);    
        saveIndividuals(logPath,hInitialPopulation, namePopulation, individualLength,config.populationSize);  

        ///*!< invokes the GPU to initialize the random trees*/
        initializePopulation<<< gridSize, blockSize >>>(dRandomTrees, nvar, individualLength, states, config.maxRandomConstant,4,config.functionRatio, config.variableRatio);    
        cudaErrorCheck("initializePopulation");    
        ///*!<return the initial population of the device to the host*/
        cudaMemcpy(hRandomTrees, dRandomTrees,sizeMemPopulation, cudaMemcpyDeviceToHost);    
        saveIndividuals(logPath,hRandomTrees, nameRandomTrees,individualLength,config.populationSize); 

        cudaEvent_t startComputeSemantics, stopComputeSemantics; /*!< This section declares and initializes the Variables for the events and captures the time elapsed in the interpretation of the initial population in the GPU*/
        cudaEventCreate(&startComputeSemantics);
        cudaEventCreate(&stopComputeSemantics);
        cudaEventRecord(startComputeSemantics);    
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeSemantics, 0, config.populationSize); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
        gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/      
     
        /*!< invokes the GPU to interpret the initial population with data train*/
        computeSemantics<<< gridSize, blockSize >>>(dInitialPopulation, uSemanticTrainCases, individualLength, uDataTrain, nrow, nvar, uPushGenes, uStackInd);
        cudaErrorCheck("computeSemantics");            
        /*!< invokes the GPU to interpret the random trees with data train*/
        computeSemantics<<< gridSize, blockSize >>>(dRandomTrees, uSemanticRandomTrees, individualLength, uDataTrain, nrow, nvar, uPushGenes, uStackInd);
        cudaErrorCheck("computeSemantics");            
        cudaEventRecord(stopComputeSemantics);
        cudaEventSynchronize(stopComputeSemantics);
        cudaEventElapsedTime(&timeComputeSemantics, startComputeSemantics, stopComputeSemantics);
        cudaEventDestroy(startComputeSemantics);
        cudaEventDestroy(stopComputeSemantics);  

        /*!< memory is deallocated for training data and auxiliary vectors for the interpreter*/
        cudaFree(uDataTrain);
        cudaFree(uStackInd);
        cudaFree(uPushGenes);  

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeError, 0, config.populationSize); 
        gridSize = (config.populationSize + blockSize - 1) / blockSize;         
        
        /*!< invokes the GPU to calculate the error (RMSE) the initial population*/
        computeError<<< gridSize, blockSize >>>(uSemanticTrainCases, uDataTrainTarget, uFit, nrow);
        cudaErrorCheck("computeError");                   
        
        
        cublasIsamin(handle, config.populationSize, uFit, incx1, &result);
        indexBestIndividual = result-1;
        
        /*!< function is necessary so that the CPU does not continue with the execution of the program and allows to capture the fitness*/
        cudaDeviceSynchronize();
        
        /*!< writing the  training fitness of the best individual on the file fitnesstrain.csv*/
        fitTraining << 0 << "," <<uFit[indexBestIndividual]<<endl;

        checkCudaErrors(cudaMallocManaged(&uSemanticTrainCasesNew,sizeMemSemanticTrain));
        checkCudaErrors(cudaMallocManaged(&uFitNew, sizeMemPopulation));
        
        cudaEvent_t startGsgp, stopGsgp;
        cudaEventCreate(&startGsgp);
        cudaEventCreate(&stopGsgp);          
        
        curandState_t* State;
        cudaMalloc((void**) &State, (twoSizePopulation) * sizeof(curandState_t));
        checkCudaErrors(cudaMallocManaged(&indexRandomTrees,twoSizeMemPopulation));
        checkCudaErrors(cudaMallocManaged(&mutationStep,sizeMemPopulation)); 
        curandState_t* statesMutationStep;
        cudaMalloc((void**) &statesMutationStep, (sizeMemPopulation) * sizeof(curandState_t));
        int index =0;   
        
        /*!< main GSGP cycle*/
        for ( int generation=1; generation<=config.numberGenerations; generation++){

            /*!< register execution time*/
            cudaEventRecord(startGsgp);
            gridSize =0, blockSize=0;
            index = generation-1;
            
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, twoSizePopulation);
            gridSize = (twoSizePopulation + blockSize - 1) / blockSize;
            init<<<gridSize, blockSize>>>(time(NULL)*index, State); /*!< initializes the random number generator*/
            cudaErrorCheck("init");

            /*!< invokes the GPU to initialize the random positions of the random trees*/
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initializeIndexRandomTrees, 0, twoSizePopulation);
            gridSize = (twoSizePopulation + blockSize - 1) / blockSize;

            initializeIndexRandomTrees<<<gridSize,blockSize >>>( config.populationSize, indexRandomTrees, State );
            cudaErrorCheck("initializeIndexRandomTrees");

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, config.populationSize);
            gridSize = (config.populationSize + blockSize - 1) / blockSize;
            init<<<gridSize, blockSize>>>(time(NULL)*index, statesMutationStep); /*!< initializes the random number generator*/
            cudaErrorCheck("init");
            
            /*!< invokes the GPU to initialize the random positions of the random trees*/
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initializeMutationStep, 0, config.populationSize);
            gridSize = (config.populationSize + blockSize - 1) / blockSize;

            initializeMutationStep<<<gridSize,blockSize >>>(mutationStep, statesMutationStep);
            cudaErrorCheck("initializeMutationStep");

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, geometricSemanticMutation, 0, sizeElementsSemanticTrain);
            gridSize = (sizeElementsSemanticTrain + blockSize - 1) / blockSize;
            
           if((gridSize*blockSize)>sizeElementsSemanticTrain){
               blockSize = minGridSize;
               gridSize = (sizeElementsSemanticTrain + blockSize - 1) / blockSize;
            }

            /*!< geometric semantic mutation with semantic train*/
            geometricSemanticMutation<<< gridSize, blockSize >>>(uSemanticTrainCases, uSemanticRandomTrees,uSemanticTrainCasesNew,
            config.populationSize, nrow, sizeElementsSemanticTrain, generation, indexRandomTrees, vectorTraces, index, mutationStep);
            cudaErrorCheck("geometricSemanticMutation");
            
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeError, 0, config.populationSize); 
            gridSize = (config.populationSize + blockSize - 1) / blockSize;
              
            /*!< invokes the GPU to calculate the error (RMSE) the new population*/
            computeError<<< gridSize,blockSize >>>(uSemanticTrainCasesNew, uDataTrainTarget, uFitNew, nrow);
            cudaErrorCheck("computeError");
         
            cublasIsamin(handle, config.populationSize, uFitNew, incxBestOffspring, &resultBestOffspring);
            indexBestOffspring = resultBestOffspring-1;
            cublasIsamax(handle, config.populationSize, uFitNew, incxWorst, &resultWorst);
            indexWorstOffspring = resultWorst-1;

            /*!< set byte values*/
            cudaMemset(indexRandomTrees,0,twoSizeMemPopulation);
            cudaMemset(mutationStep,0,sizeMemPopulation);
            cudaDeviceSynchronize();
         
            /*!< this section performs survival by updating the semantic and fitness vectors respectively*/
            if(uFitNew[indexBestOffspring] > uFit[indexBestIndividual]){
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].firstParent = indexBestIndividual;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].secondParent = -1;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].number = indexBestIndividual;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].event = -1;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].newIndividual = indexBestIndividual;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].mark= 0;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].mutStep = 0;

                for (int i = 0; i < nrow; ++i){
                    uSemanticTrainCasesNew[indexWorstOffspring*nrow+i] = uSemanticTrainCases[indexBestIndividual*nrow+i];
                }

                uFitNew[indexWorstOffspring] = uFit[indexBestIndividual];
                tempFitnes = uFit;
                uFit = uFitNew;
                uFitNew = tempFitnes;
                tempSemantic = uSemanticTrainCases;
                uSemanticTrainCases = uSemanticTrainCasesNew;
                uSemanticTrainCasesNew = tempSemantic;

                
                indexBestIndividual = indexWorstOffspring;
            }else{
                tempFitnes = uFit;
                uFit = uFitNew;
                uFitNew = tempFitnes;
                tempSemantic = uSemanticTrainCases;
                uSemanticTrainCases = uSemanticTrainCasesNew;
                uSemanticTrainCasesNew = tempSemantic;
                indexBestIndividual = indexBestOffspring;
            }
            /*!< writing the  training fitness of the best individual on the file fitnesstrain.csv*/
            fitTraining << generation << "," <<uFit[indexBestIndividual]<<endl;
            cudaEventRecord(stopGsgp);
            cudaEventSynchronize(stopGsgp);
            cudaEventElapsedTime(&generationTime, startGsgp, stopGsgp);    
        }
        markTracesGeneration(vectorTraces, config.populationSize, config.numberGenerations,  indexBestIndividual);
        saveTrace(outputNameFiles,logPath, vectorTraces, config.numberGenerations, config.populationSize);
            
        /*!< at the end of the execution  to deallocate memory*/
        cudaFree(indexRandomTrees);
        cudaFree(vectorTraces);
        cublasDestroy(handle);
        cudaFree(dInitialPopulation);
        cudaFree(dRandomTrees);
        free(hInitialPopulation);
        free(hRandomTrees);
        cudaFree(uDataTrainTarget);
        cudaFree(uFit);
        cudaFree(uFitNew);
        cudaFree(uSemanticTrainCases);
        cudaFree(uSemanticRandomTrees);
        cudaFree(uSemanticTrainCasesNew);
        cudaFree(mutationStep);
        cudaEventRecord(stopRun);
        cudaEventSynchronize(stopRun);
        cudaEventElapsedTime(&executionTime, startRun, stopRun);

         /*!< writing the time execution for stages the algorithm*/
        times << config.populationSize
        << "," << individualLength 
        << "," << nrow 
        << "," << nvar 
        << "," << executionTime/1000
        << "," << initialitionTimePopulation/1000
        << "," << timeComputeSemantics/1000
        << "," << generationTime/1000
        <<endl;
        cudaFree(State);
        cudaFree(states);
        cudaFree(statesMutationStep);
        /*!< all device allocations are removed*/
        cudaDeviceReset();
        return 0;
    }else if (!trainFile_s.empty() && !testFile_s.empty()) {
        /*!< this section is for the case when the user wants to run the algorithm with the training and test files*/

        countInputFile(trainFile, nrow, nvar);

        countInputFile(testFile, nrowTest, nvar);

        nvar--;

        individualLength = config.maxIndividualLength; /*!< Variable that stores maximum depth for individuals */
        
        sizeMemPopulation = sizeof(float) * config.populationSize * individualLength; /*!< Variable that stores size in bytes for initial population*/
        
        twoSizeMemPopulation = sizeof(float) * (config.populationSize*2); /*!< Variable that stores twice the size in bytes of an initial population to store random numbers*/
        
        sizeMemIndividuals = sizeof(float) * config.populationSize; /*!< Variable that stores size in bytes of the number of individuals in the initial population*/
        
        twoSizePopulation = (config.populationSize*2); /*!< Variable storing twice the initial population of individuals to generate random positions*/
        
        sizeMemSemanticTrain = sizeof(float)*(config.populationSize*nrow); /*!< Variable that stores the size in bytes of semantics for the entire population with training data*/
        
        sizeMemSemanticTest = sizeof(float)*(config.populationSize*nrowTest); /*!< Variable that stores the size in bytes of semantics for the entire population with test data*/
        
        sizeMemDataTrain = sizeof(float)*(nrow*nvar); /*!< Variable that stores the size in bytes the size of the training data*/
        
        sizeMemDataTest = sizeof(float)*(nrowTest*nvar); /*!< Variable that stores the size in bytes the size of the test data*/
        
        sizeElementsSemanticTrain = (config.populationSize*nrow); /*!< Variable that stores training data elements*/
        
        sizeElementsSemanticTest = (config.populationSize*nrowTest); /*!< Variable that stores test data elements*/
        
        vectorTracesMem = (sizeof(entry_)*(config.numberGenerations*config.populationSize)); /*!< Variable that stores the size in bytes of the structure to store the survival record*/
        
        /* Check if log and data diectories exists */
        checkDirectoryPath(logPath);
        
        float executionTime = 0, initialitionTimePopulation = 0, timeComputeSemantics = 0, generationTime = 0; /*!< Variables that store the time in milliseconds between the events mark1 and mark2.*/

        std::string timeExecution1 = "_processing_time"; /*!< Variable name structure responsible for indicating the run*/
        std::string timeExecution2 = ".csv"; /*!< Variable name structure responsible for indicating the file extension*/
        timeExecution1 = logPath + outputNameFiles + timeExecution1 + timeExecution2; /*!< Variable that stores file name matching*/
        std::ofstream times(timeExecution1,ios::out); /*!< pointer to the timeExecution1 file that contains the time consumed by the different algorithm modules*/
    
        cudaEvent_t startRun, stopRun; /*!< Variable used to create a start mark and a stop mark to create events*/
        cudaEventCreate(&startRun); /*!< function that initializes the start event*/
        cudaEventCreate(&stopRun); /*!< function that initializes the stop event*/

        curandState_t* states; /*!< CUDA's random number library uses curandState_t to keep track of the seed value we will store a random state for every thread*/
        cudaMalloc((void**) &states, config.populationSize * sizeof(curandState_t)); /*!< allocate space on the GPU for the random states*/
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, config.populationSize); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/

        gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/
        init<<<gridSize, blockSize>>>(time(0), states); /*!< invoke the GPU to initialize all of the random states*/

        cudaEventRecord(startRun);     
        std::string fitnessTrain  = "_fitnestrain"; /**/
        std::string fitnessTrain2 = ".csv"; /**/
        fitnessTrain = logPath + outputNameFiles + fitnessTrain + fitnessTrain2; /**/
        std::ofstream fitTraining(fitnessTrain,ios::out); /*!< pointer to the file fitnesstrain.csv containing the training fitness of the best individual at each generation*/

        std::string fitnessTest  = "_fitnestest"; /**/
        std::string fitnessTest2 = ".csv"; /**/
        fitnessTest = logPath + outputNameFiles + fitnessTest + fitnessTest2; /**/
        std::ofstream fitTesting(fitnessTest,ios::out); /*!< pointer to the file fitnesstest.csv containing the test fitness of the best individual at each generation*/

        cublasHandle_t handle; /*!< the handle to the cuBLAS library context*/
        cublasCreate(&handle); /*!< initialized using the function and is explicitly passed to every subsequent library function call*/
    
        hInitialPopulation = (float *)malloc(sizeMemPopulation); 
        hRandomTrees = (float *)malloc(sizeMemPopulation); 
        checkCudaErrors(cudaMalloc((void **)&dRandomTrees, sizeMemPopulation)); 
        checkCudaErrors(cudaMalloc((void **)&dInitialPopulation, sizeMemPopulation));
        checkCudaErrors(cudaMallocManaged(&vectorTraces,vectorTracesMem));
        checkCudaErrors(cudaMallocManaged(&uDataTrain, sizeMemDataTrain));
        checkCudaErrors(cudaMallocManaged(&uDataTest, sizeMemDataTest));      
        checkCudaErrors(cudaMallocManaged(&uDataTrainTarget, sizeof(float)*nrow));
        checkCudaErrors(cudaMallocManaged(&uDataTestTarget, sizeof(float)*nrowTest));            
        checkCudaErrors(cudaMallocManaged(&uFit, sizeMemIndividuals));
        checkCudaErrors(cudaMallocManaged(&uFitTest, sizeMemIndividuals));    
        checkCudaErrors(cudaMallocManaged(&uSemanticTrainCases,sizeMemSemanticTrain));       
        checkCudaErrors(cudaMallocManaged(&uSemanticTestCases,sizeMemSemanticTest));       
        checkCudaErrors(cudaMallocManaged(&uSemanticRandomTrees,sizeMemSemanticTrain));      
        checkCudaErrors(cudaMallocManaged(&uSemanticTestRandomTrees,sizeMemSemanticTest));
        checkCudaErrors(cudaMalloc((void**)&uPushGenes, sizeMemIndividuals));
        checkCudaErrors(cudaMalloc((void**)&uStackInd, sizeMemPopulation));            
        
        readInpuData(trainFile, testFile, uDataTrain, uDataTest, uDataTrainTarget, uDataTestTarget, nrow, nvar, nrowTest, nvar); /*!< load set data train and test*/            
        gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/            
        
        cudaEvent_t startInitialPop, stopInitialPop; /*!< this section declares and initializes the Variables for the events and captures the time elapsed in the initialization of the initial population in the GPU*/
        cudaEventCreate(&startInitialPop);
        cudaEventCreate(&stopInitialPop);
        cudaEventRecord(startInitialPop);

        ///invokes the GPU to initialize the initial population
        initializePopulation<<< gridSize, blockSize >>>(dInitialPopulation, nvar, individualLength, states, config.maxRandomConstant,4, config.functionRatio, config.variableRatio);
        cudaErrorCheck("initializePopulation");

        cudaEventRecord(stopInitialPop);
        cudaEventSynchronize(stopInitialPop);
        cudaEventElapsedTime(&initialitionTimePopulation, startInitialPop, stopInitialPop);
        cudaEventDestroy(startInitialPop);
        cudaEventDestroy(stopInitialPop);    
        
        ///*!<return the initial population of the device to the host*/
        cudaMemcpy(hInitialPopulation, dInitialPopulation, sizeMemPopulation, cudaMemcpyDeviceToHost);    
        saveIndividuals(logPath,hInitialPopulation, namePopulation, individualLength,config.populationSize);  
        
        ///*!< invokes the GPU to initialize the random trees*/
        initializePopulation<<< gridSize, blockSize >>>(dRandomTrees, nvar, individualLength, states, config.maxRandomConstant,4,config.functionRatio, config.variableRatio);    
        cudaErrorCheck("initializePopulation");    
        
        ///*!<return the initial population of the device to the host*/
        cudaMemcpy(hRandomTrees, dRandomTrees,sizeMemPopulation, cudaMemcpyDeviceToHost);    
        saveIndividuals(logPath,hRandomTrees, nameRandomTrees,individualLength,config.populationSize);  
        
        cudaEvent_t startComputeSemantics, stopComputeSemantics; /*!< This section declares and initializes the Variables for the events and captures the time elapsed in the interpretation of the initial population in the GPU*/
        cudaEventCreate(&startComputeSemantics);
        cudaEventCreate(&stopComputeSemantics);
        cudaEventRecord(startComputeSemantics);    
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeSemantics, 0, config.populationSize); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
        gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/    
         
        /*!< invokes the GPU to interpret the initial population with data train*/
        computeSemantics<<< gridSize, blockSize >>>(dInitialPopulation, uSemanticTrainCases, individualLength, uDataTrain, nrow, nvar, uPushGenes, uStackInd);
        cudaErrorCheck("computeSemantics");            
        /*!< invokes the GPU to interpret the random trees with data train*/
        computeSemantics<<< gridSize, blockSize >>>(dRandomTrees, uSemanticRandomTrees, individualLength, uDataTrain, nrow, nvar, uPushGenes, uStackInd);
        cudaErrorCheck("computeSemantics");            
        cudaEventRecord(stopComputeSemantics);
        cudaEventSynchronize(stopComputeSemantics);
        cudaEventElapsedTime(&timeComputeSemantics, startComputeSemantics, stopComputeSemantics);
        cudaEventDestroy(startComputeSemantics);
        cudaEventDestroy(stopComputeSemantics);            
        
        /*!< invokes the GPU to interpret the initial population with data train*/
        computeSemantics<<< gridSize, blockSize >>>(dInitialPopulation, uSemanticTestCases, individualLength, uDataTest, nrowTest, nvar, uPushGenes, uStackInd);
        cudaErrorCheck("computeSemantics");
        
        /*!< invokes the GPU to interpret the random trees with data test*/
        computeSemantics<<< gridSize, blockSize >>>(dRandomTrees, uSemanticTestRandomTrees, individualLength, uDataTest, nrowTest, nvar, uPushGenes, uStackInd);
        cudaErrorCheck("computeSemantics");
        cudaDeviceSynchronize();
       
        /*!< memory is deallocated for training data and auxiliary vectors for the interpreter*/
        cudaFree(uDataTrain);
        cudaFree(uDataTest);
        cudaFree(uStackInd);
        cudaFree(uPushGenes);            
        
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeError, 0, config.populationSize); 
        gridSize = (config.populationSize + blockSize - 1) / blockSize;         
        
        /*!< invokes the GPU to calculate the error (RMSE) the initial population*/
        computeError<<< gridSize, blockSize >>>(uSemanticTrainCases, uDataTrainTarget, uFit, nrow);
        cudaErrorCheck("computeError");                   
        
        cublasIsamin(handle, config.populationSize, uFit, incx1, &result);
        indexBestIndividual = result-1;

        /*!< invokes the GPU to calculate the error (RMSE) the initial population*/
        computeError<<< gridSize, blockSize >>>(uSemanticTestCases, uDataTestTarget, uFitTest, nrowTest);    
        cudaErrorCheck("computeError");         
        /*!< function is necessary so that the CPU does not continue with the execution of the program and allows to capture the fitness*/
        cudaDeviceSynchronize();
        
        /*!< writing the  training fitness of the best individual on the file fitnesstrain.csv*/
        fitTraining << 0 << "," <<uFit[indexBestIndividual]<<endl;
        /*!< writing the  test fitness of the best individual on the file fitnesstest.csv*/
        fitTesting << 0 << "," <<uFitTest[indexBestIndividual]<<endl;              

        checkCudaErrors(cudaMallocManaged(&uSemanticTrainCasesNew,sizeMemSemanticTrain));
        checkCudaErrors(cudaMallocManaged(&uFitNew, sizeMemPopulation));
        checkCudaErrors(cudaMallocManaged(&uSemanticTestCasesNew,sizeMemSemanticTest));
        checkCudaErrors(cudaMallocManaged(&uFitTestNew, sizeMemPopulation));

        cudaEvent_t startGsgp, stopGsgp;
        cudaEventCreate(&startGsgp);
        cudaEventCreate(&stopGsgp);          
        curandState_t* State;
        cudaMalloc((void**) &State, (twoSizePopulation) * sizeof(curandState_t));
        checkCudaErrors(cudaMallocManaged(&indexRandomTrees,twoSizeMemPopulation));
        checkCudaErrors(cudaMallocManaged(&mutationStep,sizeMemPopulation)); 
        curandState_t* statesMutationStep;
        cudaMalloc((void**) &statesMutationStep, (sizeMemPopulation) * sizeof(curandState_t));
        int index =0;       
        /*!< main GSGP cycle*/
        for ( int generation=1; generation<=config.numberGenerations; generation++){
            /*!< register execution time*/
            cudaEventRecord(startGsgp);
            gridSize =0, blockSize=0;
            index = generation-1;

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, twoSizePopulation);
            gridSize = (twoSizePopulation + blockSize - 1) / blockSize;
            init<<<gridSize, blockSize>>>(time(NULL)*index, State); /*!< initializes the random number generator*/
            cudaErrorCheck("init");

            /*!< invokes the GPU to initialize the random positions of the random trees*/
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initializeIndexRandomTrees, 0, twoSizePopulation);
            gridSize = (twoSizePopulation + blockSize - 1) / blockSize;
            initializeIndexRandomTrees<<<gridSize,blockSize >>>( config.populationSize, indexRandomTrees, State );
            cudaErrorCheck("initializeIndexRandomTrees");

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, config.populationSize);
            gridSize = (config.populationSize + blockSize - 1) / blockSize;
            init<<<gridSize, blockSize>>>(time(NULL)*index, statesMutationStep); /*!< initializes the random number generator*/
            cudaErrorCheck("init");
            
            /*!< invokes the GPU to initialize the random positions of the random trees*/
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initializeMutationStep, 0, config.populationSize);
            gridSize = (config.populationSize + blockSize - 1) / blockSize;
            //printf("grid %i blocksize %i \n", gridSize, blockSize);
            initializeMutationStep<<<gridSize,blockSize >>>(mutationStep, statesMutationStep);
            cudaErrorCheck("initializeMutationStep");
            
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, geometricSemanticMutation, 0, sizeElementsSemanticTrain); 
            gridSize = (sizeElementsSemanticTrain + blockSize - 1) / blockSize;
            if((gridSize*blockSize)>sizeElementsSemanticTrain){
                blockSize = minGridSize;
                gridSize = (sizeElementsSemanticTrain + blockSize - 1) / blockSize;
            }
 
            /*!< geometric semantic mutation with semantic train*/
            geometricSemanticMutation<<< gridSize, blockSize >>>(uSemanticTrainCases, uSemanticRandomTrees,uSemanticTrainCasesNew,
            config.populationSize, nrow, sizeElementsSemanticTrain, generation, indexRandomTrees, vectorTraces, index, mutationStep);
            cudaErrorCheck("geometricSemanticMutation");
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeError, 0, config.populationSize); 

            gridSize = (config.populationSize + blockSize - 1) / blockSize;
            /*!< invokes the GPU to calculate the error (RMSE) the new population*/
            computeError<<< gridSize,blockSize >>>(uSemanticTrainCasesNew, uDataTrainTarget, uFitNew, nrow);
            cudaErrorCheck("computeError");
                        /*!< this section makes use of the isamin de cublas function to determine the position of the best individual of the new population*/
            
            cublasIsamin(handle, config.populationSize, uFitNew, incxBestOffspring, &resultBestOffspring);
            indexBestOffspring = resultBestOffspring-1;
                        /*!< this section makes use of the isamin de cublas function to determine the position of the worst individual of the new population*/
            
            cublasIsamax(handle, config.populationSize, uFitNew, incxWorst, &resultWorst);
            indexWorstOffspring = resultWorst-1;

            /*!< geometric semantic mutation with semantic test*/
            cudaOccupancyMaxPotentialBlockSize(&minGridSizeTest, &blockSizeTest, geometricSemanticMutation, 0, sizeElementsSemanticTest); 
            gridSizeTest = (sizeElementsSemanticTest + blockSizeTest - 1) / blockSizeTest;
            if((gridSizeTest*blockSizeTest)>sizeElementsSemanticTest){
                blockSize = minGridSizeTest;
                gridSize = (sizeElementsSemanticTest + blockSize - 1) / blockSize;
            }
 
            geometricSemanticMutation<<< gridSizeTest, blockSizeTest >>>(uSemanticTestCases, uSemanticTestRandomTrees,uSemanticTestCasesNew,
            config.populationSize, nrowTest, sizeElementsSemanticTest, generation, indexRandomTrees, vectorTraces,index, mutationStep);
            cudaErrorCheck("geometricSemanticMutation");
            cudaDeviceSynchronize();
           
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeTest, computeError, 0, config.populationSize); 
            gridSizeTest = (config.populationSize + blockSizeTest - 1) / blockSizeTest;
            /*!< invokes the GPU to calculate the error (RMSE) the new population*/
            computeError<<< gridSizeTest,blockSizeTest >>>(uSemanticTestCasesNew, uDataTestTarget, uFitTestNew, nrowTest);
            cudaErrorCheck("computeError");
                        /*!< set byte values*/
            cudaMemset(indexRandomTrees,0,twoSizeMemPopulation);
            cudaMemset(mutationStep,0,sizeMemPopulation);
            cudaDeviceSynchronize();

            /*!< this section performs survival by updating the semantic and fitness vectors respectively*/
            if(uFitNew[indexBestOffspring] > uFit[indexBestIndividual]){
                cudaDeviceSynchronize();
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].firstParent = indexWorstOffspring;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].secondParent = -1;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].number = indexBestIndividual;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].event = -1;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].newIndividual = -1;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].mark = 0;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].mutStep = 0;

                for (int i = 0; i < nrow; ++i){
                    uSemanticTrainCasesNew[indexWorstOffspring*nrow+i] = uSemanticTrainCases[indexBestIndividual*nrow+i];
                }

                uFitNew[indexWorstOffspring] = uFit[indexBestIndividual];
                tempFitnes = uFit;
                uFit = uFitNew;
                uFitNew = tempFitnes;
                tempSemantic = uSemanticTrainCases;
                uSemanticTrainCases = uSemanticTrainCasesNew;
                uSemanticTrainCasesNew = tempSemantic;
                
                for (int j = 0; j < nrowTest; ++j){
                    uSemanticTestCasesNew[indexWorstOffspring*nrowTest+j] = uSemanticTestCases[indexBestIndividual*nrowTest+j];
                }
                uFitTestNew[indexWorstOffspring] = uFitTest[indexBestIndividual];

                tempFitnesTest = uFitTest;
                uFitTest = uFitTestNew;
                uFitTestNew = tempFitnesTest;
                tempSemanticTest = uSemanticTestCases;
                uSemanticTestCases = uSemanticTestCasesNew;
                uSemanticTestCasesNew = tempSemanticTest;
                indexBestIndividual = indexWorstOffspring;
            }else{

                tempFitnes = uFit;
                uFit = uFitNew;
                uFitNew = tempFitnes;
                tempSemantic = uSemanticTrainCases;
                uSemanticTrainCases = uSemanticTrainCasesNew;
                uSemanticTrainCasesNew = tempSemantic;
                tempFitnesTest = uFitTest;
                uFitTest = uFitTestNew;
                uFitTestNew = tempFitnesTest;
                tempSemanticTest = uSemanticTestCases;
                uSemanticTestCases = uSemanticTestCasesNew;
                uSemanticTestCasesNew = tempSemanticTest;
                indexBestIndividual = indexBestOffspring;
            }
            /*!< writing the  training fitness of the best individual on the file fitnesstrain.csv*/
            fitTraining << generation << ","<<uFit[indexBestIndividual]<<endl;
            /*!< writing the  test fitness of the best individual on the file fitnesstest.csv*/
            fitTesting << generation << ","<<uFitTest[indexBestIndividual]<<endl;
            
            cudaEventRecord(stopGsgp);
            cudaEventSynchronize(stopGsgp);
            cudaEventElapsedTime(&generationTime, startGsgp, stopGsgp);    
        }
        cudaDeviceSynchronize();
        markTracesGeneration(vectorTraces, config.populationSize, config.numberGenerations,  indexBestIndividual);
        saveTrace(outputNameFiles,logPath, vectorTraces, config.numberGenerations, config.populationSize);
            
        /*!< at the end of the execution  to deallocate memory*/
        cudaFree(indexRandomTrees);
        cudaFree(vectorTraces);
        cudaFree(State);
        cublasDestroy(handle);
        cudaFree(dInitialPopulation);
        cudaFree(dRandomTrees);
        free(hInitialPopulation);
        free(hRandomTrees);
        cudaFree(uDataTrainTarget);
        cudaFree(uDataTestTarget);
        cudaFree(uFit);
        cudaFree(uFitNew);
        cudaFree(uSemanticTrainCases);
        cudaFree(uSemanticRandomTrees);
        cudaFree(uSemanticTrainCasesNew);
        cudaFree(uSemanticTestCases);
        cudaFree(uSemanticTestRandomTrees);
        cudaFree(uSemanticTestCasesNew);     
        cudaFree(uFitTest);
        cudaFree(uFitTestNew);
        cudaFree(mutationStep);
        cudaEventRecord(stopRun);
        cudaEventSynchronize(stopRun);
        cudaEventElapsedTime(&executionTime, startRun, stopRun);

        /*!< writing the time execution for stages the algorithm*/
        times << config.populationSize
        << "," << individualLength 
        << "," << nrow 
        << "," << nvar 
        << "," << executionTime/1000
        << "," << initialitionTimePopulation/1000
        << "," << timeComputeSemantics/1000
        << "," << generationTime/1000
        <<endl;
        cudaFree(states);
        cudaFree(statesMutationStep);
        /*!< all device allocations are removed*/
        cudaDeviceReset();
            
    }if (!la.empty()) {

        countInputFile(path_test, nrowTest, nvar);
        nvar--;

        namePopulation = la + namePopulation;
        nameRandomTrees = la + nameRandomTrees;
        std::string outFile (pathOutFile);
        outFile = logPath + outFile;
        outFile.c_str();
        individualLength = config.maxIndividualLength; /*!< Variable that stores maximum depth for individuals */
        sizeMemPopulation = sizeof(float) * config.populationSize * individualLength; /*!< Variable that stores size in bytes for initial population*/
        sizeMemIndividuals = sizeof(float) * config.populationSize; /*!< Variable that stores size in bytes of the number of individuals in the initial population*/

        float *initPopulation, *randomTress, *dInitialPopulation,*dRandomTrees; /*!< This vector pointers to store the individuals of the initial population and random trees */
        initPopulation = (float*)malloc(sizeMemPopulation); /*!<  Variable that stores the size in bytes the initial population */
        randomTress = (float*)malloc(sizeMemPopulation);  /*!< Variable that stores the size in bytes the initial population */

        checkCudaErrors(cudaMalloc((void **)&dRandomTrees, sizeMemPopulation)); 
        checkCudaErrors(cudaMalloc((void **)&dInitialPopulation, sizeMemPopulation));

        readPopulation(initPopulation, randomTress, config.populationSize, individualLength, logPath, namePopulation, nameRandomTrees);

        ///*!<return the initial population of the device to the host*/
        cudaMemcpy(dInitialPopulation, initPopulation, sizeMemPopulation, cudaMemcpyHostToDevice); 
        cudaMemcpy(dRandomTrees, randomTress, sizeMemPopulation, cudaMemcpyHostToDevice); 

        int sizeDataTest = sizeof(float)*(nrowTest*nvar); /*!< Variable that stores the size in bytes the size of the test data*/
        int sizeDataTestTarget = sizeof(float)*(nrowTest); /*!< Variable that stores the size in bytes the size of the target data */
        float *unssenDataTest, *dUnssenDataTest, *unssenDataTestTarget; /*!< This vector pointers to store the individuals of the test data and target data */
        unssenDataTest = (float *)malloc(sizeDataTest); /*!< Reserve memory on host*/
        unssenDataTestTarget = (float *)malloc(sizeDataTestTarget); /*!< Reserve memory on host*/
        checkCudaErrors(cudaMalloc((void **)&dUnssenDataTest, sizeDataTest));

        readInpuTestData(path_test, unssenDataTest, unssenDataTestTarget, nrowTest, nvar);

        cudaMemcpy(dUnssenDataTest, unssenDataTest, sizeDataTest, cudaMemcpyHostToDevice); 
        
        sizeMemSemanticTest = sizeof(float)*(config.populationSize*nrowTest); /*!< Variable that stores the size in bytes of semantics for the entire population with test data*/

        checkCudaErrors(cudaMalloc((void**)&uPushGenes, sizeMemIndividuals));
        checkCudaErrors(cudaMalloc((void**)&uStackInd, sizeMemPopulation));  

        float *uSemanticCases, *hSemanticCases, *uSemanticRandomTrees,*hSemanticRandomTrees; /*!< pointer of vectors that contain the semantics of an individual in the population, calculated with the training set and test in generation g and its allocation in GPU*/
        checkCudaErrors(cudaMalloc((void**)&uSemanticCases,sizeMemSemanticTest));            
        checkCudaErrors(cudaMalloc((void**)&uSemanticRandomTrees,sizeMemSemanticTest));   
        hSemanticCases = (float*)malloc(sizeMemSemanticTest);
        hSemanticRandomTrees= (float*)malloc(sizeMemSemanticTest);             

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeSemantics, 0, config.populationSize); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
        gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/            

        /*!< invokes the GPU to interpret the initial population with data train*/
        computeSemantics<<< gridSize, blockSize >>>(dInitialPopulation, uSemanticCases, individualLength, dUnssenDataTest, nrowTest, nvar, uPushGenes, uStackInd);
        cudaErrorCheck("computeSemantics");
        cudaMemcpy(hSemanticCases,uSemanticCases, sizeMemSemanticTest,cudaMemcpyDeviceToHost);

        computeSemantics<<< gridSize, blockSize >>>(dRandomTrees, uSemanticRandomTrees, individualLength, dUnssenDataTest, nrowTest, nvar, uPushGenes, uStackInd);
        cudaErrorCheck("computeSemantics");
        cudaMemcpy(hSemanticRandomTrees,uSemanticRandomTrees, sizeMemSemanticTest,cudaMemcpyDeviceToHost);

        /*!< Create file for saved results of best model with the unseen data*/
        std::ofstream OUT(outFile,ios::out);
        
        evaluate_data(pathTrace, config.numberGenerations, hSemanticCases, hSemanticRandomTrees, OUT, config.logPath, nrowTest, nvar);
        
        
        free(unssenDataTest); 
        free(unssenDataTestTarget);
        free(initPopulation);
        free(randomTress);
        cudaFree(dInitialPopulation);
        cudaFree(dRandomTrees);
        cudaFree(uSemanticCases);
        cudaFree(uSemanticRandomTrees);
        cudaFree(uPushGenes);
        cudaFree(uStackInd);
    }
    cudaDeviceReset();
    return 0;
}