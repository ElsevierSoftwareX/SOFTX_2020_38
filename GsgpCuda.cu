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
using namespace std;   

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
    
    srand(time(NULL)); /*!< Initialization of the seed for the generation of random numbers*/

    cudaSetDevice(0); /*!< Select a GPU device*/
 
    char output_model[50]=""; /*!< Name of output files*/
    for (int i=1; i<argc-1; i++){
        if(strncmp(argv[i],"-output_model",10) == 0) {
            strcat(output_model,argv[++i]);
        }      
    }

    std::string outputNameFiles(output_model); /* name of file for save the output files */

    printf("\n Starting GsgpCuda \n\n");
    
    readConfigFile(&config); /*!< reading the parameters of the algorithm */
    
    const int sizeMaxDepthIndividual = (int)exp2(config.maxDepth*1.0) - 1; /*!< variable that stores maximum depth for individuals */
    
    int sizeMemPopulation = sizeof(float) * config.populationSize * sizeMaxDepthIndividual; /*!< variable that stores size in bytes for initial population*/
    
    int twoSizeMemPopulation = sizeof(float) * (config.populationSize*2); /*!< variable that stores twice the size in bytes of an initial population to store random numbers*/
    
    int sizeMemIndividuals = sizeof(float) * config.populationSize; /*!< variable that stores size in bytes of the number of individuals in the initial population*/
    
    long int twoSizePopulation = (config.populationSize*2); /*!< variable storing twice the initial population of individuals to generate random positions*/
    
    long int sizeMemSemanticTrain = sizeof(float)*(config.populationSize*config.nrow); /*!< variable that stores the size in bytes of semantics for the entire population with training data*/
    
    long int sizeMemSemanticTest = sizeof(float)*(config.populationSize*config.nrowTest); /*!< variable that stores the size in bytes of semantics for the entire population with test data*/
    
    long int sizeMemDataTrain = sizeof(float)*(config.nrow*config.nvar); /*!< variable that stores the size in bytes the size of the training data*/
    
    long int sizeMemDataTest = sizeof(float)*(config.nrowTest*config.nvarTest); /*!< variable that stores the size in bytes the size of the test data*/
    
    long int sizeElementsSemanticTrain = (config.populationSize*config.nrow); /*!< variable that stores training data elements*/
    
    long int sizeElementsSemanticTest = (config.populationSize*config.nrowTest); /*!< variable that stores test data elements*/
    
    size_t structMemMutation = (sizeof(entry_)*config.populationSize); /*!< variable that stores the size in bytes of the structure to store the mutation record*/
    
    size_t structMemSurvivor = (sizeof(entry_)*config.maxNumberGenerations); /*!< variable that stores the size in bytes of the structure to store the survival record*/
    
    long int vectorTracesMem = (sizeof(entry_)*config.maxNumberGenerations*config.populationSize); /*!< variable that stores the size in bytes of the structure to store the survival record*/

    int gridSize,minGridSize,blockSize; /*!< variables that store the execution configuration for a kernel in the GPU*/
    
    int gridSizeTest,minGridSizeTest,blockSizeTest; /*!< variables that store the execution configuration for a kernel in the GPU*/
    
    std::string logPath (config.logPath); /* Path of directory for data files and log files generated in execution */

    std::string dataPath (config.dataPath); /* Path of directory for data files for training */

    std::string dataPathTest (config.dataPathTest); /* Path of directory for data files for test */

    std::string namePopulation = "_initialPopulation"; /*!< Name of file for save the initial population  */

    namePopulation = outputNameFiles + namePopulation;

    std::string nameRandomTrees = "_randomTrees"; /*!< name of file for save the random trees */

    nameRandomTrees = outputNameFiles + nameRandomTrees;
    
    if (argc>3) {

        char pathTrace[50]=""; /*!< Name of the file trace of best model*/
        for (int i=1; i<argc-1; i++){
            if(strncmp(argv[i],"-model",10) == 0) {
                strcat(pathTrace,argv[++i]);
            }      
        }
        std::string tmp(pathTrace);
        namePopulation = tmp + namePopulation;
        nameRandomTrees = tmp + nameRandomTrees;

        char path_test[50]=""; /*!< Name of the file with test instances*/
        for (int i=1; i<argc-1; i++){
            if(strncmp(argv[i],"-input_data",10) == 0) {
                strcat(path_test,argv[++i]);
            }      
        }
        
        char pathOutFile[50]=""; /*!< Name of the file to output values*/
        for (int i=1; i<argc-1; i++){
            if(strncmp(argv[i],"-prediction_output",10) == 0) {
                strcat(pathOutFile,argv[++i]);
            }      
        }

        std::string outFile (pathOutFile);
        
        int sizeDataTest = sizeof(float)*(config.nrowTest*config.nvarTest); /*!< variable that stores the size in bytes the size of the test data*/
        int sizeDataTestTarget = sizeof(float)*(config.nrowTest); /*!< variable that stores the size in bytes the size of the target data */
        float *unssenDataTest, *unssenDataTestTarget; /*!< This vector pointers to store the individuals of the test data and target data */
        unssenDataTest = (float *)malloc(sizeDataTest); /*!< Reserve memory on host*/
        unssenDataTestTarget = (float *)malloc(sizeDataTestTarget); /*!< Reserve memory on host*/

        readInpuTestData(path_test, unssenDataTest, unssenDataTestTarget, config.nrowTest, config.nvarTest);
        
        float *initPopulation, *randomTress; /*!< This vector pointers to store the individuals of the initial population and random trees */
        initPopulation = (float*)malloc(sizeMemPopulation); /*!<  Variable that stores the size in bytes the initial population */
        randomTress = (float*)malloc(sizeMemPopulation);  /*!< Variable that stores the size in bytes the initial population */
        
        readPopulation(initPopulation, randomTress, config.populationSize, sizeMaxDepthIndividual, logPath, namePopulation, nameRandomTrees);
        
        /*!< Create file for saved results of best model with the unseen data*/
        std::ofstream OUT(outFile,ios::out);
        
        //function to evaluate new data with the best model
        evaluate_unseen_new_data(pathTrace, config.maxNumberGenerations, sizeMaxDepthIndividual, initPopulation, randomTress, OUT, config.logPath, unssenDataTest, config.nrowTest, config.populationSize, config.nvarTest);
	    
        free(unssenDataTest); 
        free(unssenDataTestTarget);
        free(initPopulation);
        free(randomTress);
    }else {
        /* Check if log and data diectories exists */
        checkDirectoryPath(logPath);
        checkDirectoryPath(dataPath);
        checkDirectoryPath(dataPathTest);

        std::vector<string> files = vector<string>(); /**/
        std::vector<string> filesTest = vector<string>(); /**/
        
        /* Get list of data files in data directory */
        list_dir(dataPath, config.trainFile, config.useMultipleTrainFiles, files);
        list_dir(dataPathTest, config.testFile, config.useMultipleTrainFiles, filesTest);
 
        std::string timeExecution1 = "_processing_time"; /*!< variable name structure responsible for indicating the run*/
 
        std::string timeExecution2 = ".csv"; /*!< variable name structure responsible for indicating the file extension*/
 
        //std::string dateTime =currentDateTime(); /*!< variable name structure responsible for capturing the date and time of the run*/
 
        timeExecution1 = logPath + outputNameFiles + timeExecution1 + timeExecution2; /*!< variable that stores file name matching*/
 
        std::ofstream times(timeExecution1,ios::out); /*!< pointer to the timeExecution1 file that contains the time consumed by the different algorithm modules*/
 
        float executionTime = 0, initialitionTimePopulation = 0, timeComputeSemantics = 0, generationTime = 0; /*!< variables that store the time in milliseconds between the events mark1 and mark2.*/

        /*!< algorithm run cycle*/
        for (int runs_ = 0; runs_ < config.numberRuns; runs_++){

            executionTime = 0, initialitionTimePopulation = 0, timeComputeSemantics = 0, generationTime = 0; /*!< variables that store the time in milliseconds between the events mark1 and mark2.*/    

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
            //std::string fitnessTrain3 = currentDateTime(); /**/
            fitnessTrain = logPath + outputNameFiles + fitnessTrain + fitnessTrain2; /**/

            std::ofstream fitTraining(fitnessTrain,ios::out); /*!< pointer to the file fitnesstrain.csv containing the training fitness of the best individual at each generation*/

            std::string fitnessTest  = "_fitnestest"; /**/
            std::string fitnessTest2 = ".csv"; /**/
            //std::string fitnessTest3 = currentDateTime(); /**/
            fitnessTest = logPath + outputNameFiles + fitnessTest + fitnessTest2; /**/

            std::ofstream fitTesting(fitnessTest,ios::out); /*!< pointer to the file fitnesstest.csv containing the test fitness of the best individual at each generation*/

            cublasHandle_t handle; /*!< the handle to the cuBLAS library context*/

            cublasCreate(&handle); /*!< initialized using the function and is explicitly passed to every subsequent library function call*/
 
            float *dInitialPopulation,*dRandomTrees,*hInitialPopulation,*hRandomTrees;  /*!< This block contains the vectors of pointers to store the population and random trees and space allocation in the GPU*/
            hInitialPopulation = (float *)malloc(sizeMemPopulation); 
            hRandomTrees = (float *)malloc(sizeMemPopulation); 
            checkCudaErrors(cudaMalloc((void **)&dRandomTrees, sizeMemPopulation)); 
            checkCudaErrors(cudaMalloc((void **)&dInitialPopulation, sizeMemPopulation));

            entry  *dStructMutation, *dStructMutationTest ,*dStructSurvivor,*vectorTraces,*vectorTracesTest; /*!< This block contains the vectors of pointers to store the structure to keep track of mutation and survival and space allocation in the GPU*/
            checkCudaErrors(cudaMallocManaged(&dStructMutation,structMemMutation));
            checkCudaErrors(cudaMallocManaged(&dStructMutationTest,structMemMutation)); 
            checkCudaErrors(cudaMallocManaged(&dStructSurvivor,structMemSurvivor));
            checkCudaErrors(cudaMallocManaged(&vectorTraces,vectorTracesMem));
            checkCudaErrors(cudaMallocManaged(&vectorTracesTest,vectorTracesMem));

            std::string dataFile = (files[runs_]); /**/
            std::string dataFileTest = (filesTest[runs_]); /**/
            char pathTrain[50]=""; /*!< name of the file with training instances */
            char pathTest[50]="";  /*!< name of the file with test instances*/
 
            strcpy(pathTrain,config.dataPath); /*!< name of the file with training instances */
            strcat(pathTrain,dataFile.c_str());
 
            strcpy(pathTest,config.dataPathTest); 
            strcat(pathTest,dataFileTest.c_str());        

            float *uDataTrain, *uDataTest, *uDataTrainTarget, *uDataTestTarget;  /*!< this block contains the pointer of vectors for the input data and target values ​​and assignment in the GPU*/
            checkCudaErrors(cudaMallocManaged(&uDataTrain, sizeMemDataTrain));
            checkCudaErrors(cudaMallocManaged(&uDataTest, sizeMemDataTest));      
            checkCudaErrors(cudaMallocManaged(&uDataTrainTarget, sizeof(float)*config.nrow));
            checkCudaErrors(cudaMallocManaged(&uDataTestTarget, sizeof(float)*config.nrowTest));            

            float *uFit, *uFitTest; /*!< pointers of vectors of training and test fitness values at generation g and assignment in the GPU*/
            checkCudaErrors(cudaMallocManaged(&uFit, sizeMemIndividuals));
            checkCudaErrors(cudaMallocManaged(&uFitTest, sizeMemIndividuals));    

            float *uSemanticTrainCases, *uSemanticTestCases, *uSemanticRandomTrees, *uSemanticTestRandomTrees; /*!< pointer of vectors that contain the semantics of an individual in the population, calculated with the training set and test in generation g and its allocation in GPU*/
            checkCudaErrors(cudaMallocManaged(&uSemanticTrainCases,sizeMemSemanticTrain));       
            checkCudaErrors(cudaMallocManaged(&uSemanticTestCases,sizeMemSemanticTest));       
            checkCudaErrors(cudaMallocManaged(&uSemanticRandomTrees,sizeMemSemanticTrain));      
            checkCudaErrors(cudaMallocManaged(&uSemanticTestRandomTrees,sizeMemSemanticTest));             

            float *uStackInd; /*!< auxiliary pointer vectors for the interpreter and calculate the semantics for the populations and assignment in the GPU*/
            int   *uPushGenes;
            checkCudaErrors(cudaMalloc((void**)&uPushGenes, sizeMemIndividuals));
            checkCudaErrors(cudaMalloc((void**)&uStackInd, sizeMemPopulation));            
            float *tempSemantic,*tempFitnes,*tempSemanticTest,*tempFitnesTest; /*!< temporal variables to perform the movement of pointers in survival*/

            readInpuData(pathTrain, pathTest, uDataTrain, uDataTest, uDataTrainTarget, uDataTestTarget, config.nrow, config.nvar, config.nrowTest, config.nvarTest); /*!< load set data train and test*/            

            gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/            
            cudaEvent_t startInitialPop, stopInitialPop; /*!< this section declares and initializes the variables for the events and captures the time elapsed in the initialization of the initial population in the GPU*/
            cudaEventCreate(&startInitialPop);
            cudaEventCreate(&stopInitialPop);
            cudaEventRecord(startInitialPop);    
            ///*!< invokes the GPU to initialize the initial population*/
            initializePopulation<<< gridSize, blockSize >>>(dInitialPopulation, config.nvar, sizeMaxDepthIndividual, states, config.maxRandomConstant,4);
            cudaErrorCheck("initializePopulation");    
            cudaEventRecord(stopInitialPop);
            cudaEventSynchronize(stopInitialPop);
            cudaEventElapsedTime(&initialitionTimePopulation, startInitialPop, stopInitialPop);
            cudaEventDestroy(startInitialPop);
            cudaEventDestroy(stopInitialPop);    
            ///*!<return the initial population of the device to the host*/
            cudaMemcpy(hInitialPopulation, dInitialPopulation, sizeMemPopulation, cudaMemcpyDeviceToHost);    
            saveIndividuals(logPath,hInitialPopulation, namePopulation, sizeMaxDepthIndividual,config.populationSize);  
            ///*!< invokes the GPU to initialize the random trees*/
            initializePopulation<<< gridSize, blockSize >>>(dRandomTrees, config.nvar, sizeMaxDepthIndividual, states, config.maxRandomConstant,4);    
            cudaErrorCheck("initializePopulation");    
            ///*!<return the initial population of the device to the host*/
            cudaMemcpy(hRandomTrees, dRandomTrees,sizeMemPopulation, cudaMemcpyDeviceToHost);    
            saveIndividuals(logPath,hRandomTrees, nameRandomTrees,sizeMaxDepthIndividual,config.populationSize);  
            cudaEvent_t startComputeSemantics, stopComputeSemantics; /*!< This section declares and initializes the variables for the events and captures the time elapsed in the interpretation of the initial population in the GPU*/
            cudaEventCreate(&startComputeSemantics);
            cudaEventCreate(&stopComputeSemantics);
            cudaEventRecord(startComputeSemantics);    
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeSemantics, 0, config.populationSize); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
            gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/            
            /*!< invokes the GPU to interpret the initial population with data train*/
            computeSemantics<<< gridSize, blockSize >>>(dInitialPopulation, uSemanticTrainCases, sizeMaxDepthIndividual, uDataTrain, config.nrow, config.nvar, uPushGenes, uStackInd);
            cudaErrorCheck("computeSemantics");            
            /*!< invokes the GPU to interpret the random trees with data train*/
            computeSemantics<<< gridSize, blockSize >>>(dRandomTrees, uSemanticRandomTrees, sizeMaxDepthIndividual, uDataTrain, config.nrow, config.nvar, uPushGenes, uStackInd);
            cudaErrorCheck("computeSemantics");            
            cudaEventRecord(stopComputeSemantics);
            cudaEventSynchronize(stopComputeSemantics);
            cudaEventElapsedTime(&timeComputeSemantics, startComputeSemantics, stopComputeSemantics);
            cudaEventDestroy(startComputeSemantics);
            cudaEventDestroy(stopComputeSemantics);            
            /*!< invokes the GPU to interpret the initial population with data train*/
            computeSemantics<<< gridSize, blockSize >>>(dInitialPopulation, uSemanticTestCases, sizeMaxDepthIndividual, uDataTest, config.nrowTest, config.nvarTest, uPushGenes, uStackInd);
            cudaErrorCheck("computeSemantics");           
            /*!< invokes the GPU to interpret the random trees with data test*/
            computeSemantics<<< gridSize, blockSize >>>(dRandomTrees, uSemanticTestRandomTrees, sizeMaxDepthIndividual, uDataTest, config.nrowTest, config.nvarTest, uPushGenes, uStackInd);
            cudaErrorCheck("computeSemantics");            
            /*!< memory is deallocated for training data and auxiliary vectors for the interpreter*/
            cudaFree(uDataTrain);
            cudaFree(uDataTest);
            cudaFree(uStackInd);
            cudaFree(uPushGenes);            
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeError, 0, config.populationSize); 
            gridSize = (config.populationSize + blockSize - 1) / blockSize;         
            
            /*!< invokes the GPU to calculate the error (RMSE) the initial population*/
            computeError<<< gridSize, blockSize >>>(uSemanticTrainCases, uDataTrainTarget, uFit, config.nrow);
            cudaErrorCheck("computeError");                   
            
            int result,incx1=1,indexBestIndividual; /*!< this section makes use of the isamin de cublas function to determine the position of the best individual*/
            cublasIsamin(handle, config.populationSize, uFit, incx1, &result);
            indexBestIndividual = result-1;

            /*!< invokes the GPU to calculate the error (RMSE) the initial population*/
            computeError<<< gridSize, blockSize >>>(uSemanticTestCases, uDataTestTarget, uFitTest, config.nrowTest);    
            cudaErrorCheck("computeError");         
            /*!< function is necessary so that the CPU does not continue with the execution of the program and allows to capture the fitness*/
            cudaDeviceSynchronize();
            
            /*!< writing the  training fitness of the best individual on the file fitnesstrain.csv*/
            fitTraining << runs_ << "," << 0 << "," << uFit[indexBestIndividual]<<endl;
            /*!< writing the  test fitness of the best individual on the file fitnesstest.csv*/
            fitTesting  << runs_ << "," << 0 << "," << uFitTest[indexBestIndividual]<<endl;              

            float *uSemanticTrainCasesNew, *uFitNew, *uSemanticTestCasesNew, *uFitTestNew; /*!< vectors that contain the semantics of an individual in the population, calculated in the training and test set in the g + 1 generation and its allocation in GPU*/
            checkCudaErrors(cudaMallocManaged(&uSemanticTrainCasesNew,sizeMemSemanticTrain));
            checkCudaErrors(cudaMallocManaged(&uFitNew, sizeMemPopulation));
            checkCudaErrors(cudaMallocManaged(&uSemanticTestCasesNew,sizeMemSemanticTest));
            checkCudaErrors(cudaMallocManaged(&uFitTestNew, sizeMemPopulation));

            cudaEvent_t startGsgp, stopGsgp;
            cudaEventCreate(&startGsgp);
            cudaEventCreate(&stopGsgp);          
            curandState_t* State;
            cudaMalloc((void**) &State, (twoSizePopulation) * sizeof(curandState_t));

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init, 0, twoSizePopulation);
            gridSize = (twoSizePopulation + blockSize - 1) / blockSize;
            init<<<gridSize, blockSize>>>(time(NULL), State);         

            float *indexRandomTrees; /*!< vector of pointers to save random positions of random trees and allocation in GPU*/
            checkCudaErrors(cudaMallocManaged(&indexRandomTrees,twoSizeMemPopulation));         
            /*!< main GSGP cycle*/
            for ( int generation=1; generation<=config.maxNumberGenerations; generation++){

                /*!< register execution time*/
                cudaEventRecord(startGsgp);
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initializeIndexRandomTrees, 0, twoSizePopulation);
                gridSize = (twoSizePopulation + blockSize - 1) / blockSize;
     
                initializeIndexRandomTrees<<<gridSize,blockSize >>>( config.populationSize, indexRandomTrees, State );
                cudaErrorCheck("initializeIndexRandomTrees");

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, geometricSemanticMutation, 0, sizeElementsSemanticTrain); 
                gridSize = (sizeElementsSemanticTrain + blockSize - 1) / blockSize;
                /*!< geometric semantic mutation with semantic train*/
                geometricSemanticMutation<<< gridSize, blockSize >>>(uSemanticTrainCases, uSemanticRandomTrees,uSemanticTrainCasesNew,
                    config.populationSize, config.nrow, sizeElementsSemanticTrain, generation, indexRandomTrees, dStructMutation, vectorTraces);
                cudaErrorCheck("geometricSemanticMutation");
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeError, 0, config.populationSize); 


                gridSize = (config.populationSize + blockSize - 1) / blockSize;
                /*!< invokes the GPU to calculate the error (RMSE) the new population*/
                computeError<<< gridSize,blockSize >>>(uSemanticTrainCasesNew, uDataTrainTarget, uFitNew, config.nrow);
                cudaErrorCheck("computeError");
         
                /*!< this section makes use of the isamin de cublas function to determine the position of the best individual of the new population*/
                int resultBestOffspring,incxBestOffspring=1,indexBestOffspring;
                cublasIsamin(handle, config.populationSize, uFitNew, incxBestOffspring, &resultBestOffspring);
                indexBestOffspring = resultBestOffspring-1;
         
                /*!< this section makes use of the isamin de cublas function to determine the position of the worst individual of the new population*/
                int resultWorst,incxWorst=1,indexWorstOffspring;
                cublasIsamax(handle, config.populationSize, uFitNew, incxWorst, &resultWorst);
                indexWorstOffspring = resultWorst-1;

                /*!< geometric semantic mutation with semantic test*/
                cudaOccupancyMaxPotentialBlockSize(&minGridSizeTest, &blockSizeTest, geometricSemanticMutation, 0, sizeElementsSemanticTest); 
                gridSizeTest = (sizeElementsSemanticTest + blockSizeTest - 1) / blockSizeTest;
         
                geometricSemanticMutation<<< gridSizeTest, blockSizeTest >>>(uSemanticTestCases, uSemanticTestRandomTrees,uSemanticTestCasesNew,
                config.populationSize, config.nrowTest, sizeElementsSemanticTest, generation, indexRandomTrees, dStructMutationTest, vectorTracesTest);
                cudaErrorCheck("geometricSemanticMutation");

                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeTest, computeError, 0, config.populationSize); 
                gridSizeTest = (config.populationSize + blockSizeTest - 1) / blockSizeTest;
                /*!< invokes the GPU to calculate the error (RMSE) the new population*/
                computeError<<< gridSizeTest,blockSizeTest >>>(uSemanticTestCasesNew, uDataTestTarget, uFitTestNew, config.nrowTest);
                cudaErrorCheck("computeError");
         
                /*!< set byte values*/
                cudaMemset(indexRandomTrees,0,twoSizeMemPopulation);
                cudaDeviceSynchronize();
         
                /*!< this section performs survival by updating the semantic and fitness vectors respectively*/
                int index = generation-1;
                if(uFitNew[indexBestOffspring] > uFit[indexBestIndividual]){
                    for (int i = 0; i < config.nrow; ++i){
                        uSemanticTrainCasesNew[indexWorstOffspring*config.nrow+i] = uSemanticTrainCases[indexBestIndividual*config.nrow+i];
                    }

                uFitNew[indexWorstOffspring] = uFit[indexBestIndividual];
         
                    tempFitnes = uFit;
                    uFit = uFitNew;
                    uFitNew = tempFitnes;
                    tempSemantic = uSemanticTrainCases;
                    uSemanticTrainCases = uSemanticTrainCasesNew;
                    uSemanticTrainCasesNew = tempSemantic;
               
                    for (int j = 0; j < config.nrowTest; ++j){
                        uSemanticTestCasesNew[indexWorstOffspring*config.nrowTest+j] = uSemanticTestCases[indexBestIndividual*config.nrowTest+j];
                    }

                    uFitTestNew[indexWorstOffspring] = uFitTest[indexBestIndividual];
                    vectorTraces[(index*config.populationSize)+indexWorstOffspring].firstParent = indexBestIndividual;
                    vectorTraces[(index*config.populationSize)+indexWorstOffspring].secondParent = -1;
                    vectorTraces[(index*config.populationSize)+indexWorstOffspring].number=indexBestIndividual;
                    vectorTraces[(index*config.populationSize)+indexWorstOffspring].event = -1;
                    vectorTraces[(index*config.populationSize)+indexWorstOffspring].newIndividual = indexBestIndividual;
                    vectorTraces[(index*config.populationSize)+indexWorstOffspring].mark=0;
                    vectorTraces[(index*config.populationSize)+indexWorstOffspring].mutStep = 0;

                    tempFitnesTest = uFitTest;
                    uFitTest = uFitTestNew;
                    uFitTestNew = tempFitnesTest;
                    tempSemanticTest = uSemanticTestCases;
                    uSemanticTestCases = uSemanticTestCasesNew;
                    uSemanticTestCasesNew = tempSemanticTest;
                    indexBestIndividual = indexWorstOffspring;
                }else
                {
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
                fitTraining<< runs_ << "," << generation << ","<<uFit[indexBestIndividual]<<endl;
                /*!< writing the  test fitness of the best individual on the file fitnesstest.csv*/
                fitTesting<< runs_ << "," << generation << ","<<uFitTest[indexBestIndividual]<<endl;
                cudaEventRecord(stopGsgp);
                cudaEventSynchronize(stopGsgp);
                cudaEventElapsedTime(&generationTime, startGsgp, stopGsgp);    
            }
            //saveTraceComplete(logPath, vectorTraces, config.maxNumberGenerations, config.populationSize);
            
            markTracesGeneration(vectorTraces, config.populationSize, config.maxNumberGenerations,  indexBestIndividual);

            saveTrace(outputNameFiles,logPath, vectorTraces, config.maxNumberGenerations, config.populationSize);
            
            /*!< at the end of the execution  to deallocate memory*/
            cudaFree(indexRandomTrees);
            cudaFree(dStructMutation);
            cudaFree(dStructSurvivor);
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
            cudaEventRecord(stopRun);
            cudaEventSynchronize(stopRun);
            cudaEventElapsedTime(&executionTime, startRun, stopRun);

            /*!< writing the time execution for stages the algorithm*/
            times << runs_
            << "," << config.populationSize
            << "," << sizeMaxDepthIndividual 
            << "," << config.nrow 
            << "," << config.nvar 
            << "," << executionTime/1000
            << "," << initialitionTimePopulation/1000
            << "," << timeComputeSemantics/1000
            << "," << generationTime/1000
            <<endl;
            cudaFree(states);
            /*!< all device allocations are removed*/
            cudaDeviceReset();
        } 
     }
    return 0;
}

