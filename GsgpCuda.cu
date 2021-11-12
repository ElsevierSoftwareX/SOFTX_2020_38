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
void validacion(std::string name, float *targer, int nrow){
    
    std::fstream in(name.c_str(),ios::in);
    char Str[1024];
    float RMSE=0;
    float t=0;
    float tmp[nrow];
    if (!in.is_open())
    {
      cout<<endl<<"ERROR: TRAINING FILE NOT FOUND." << endl;
      exit(-1);
    }

    for(int i=0;i<nrow;i++){
        in>>Str;
        tmp[i]=atof(Str);
        RMSE += (targer[i]-tmp[i])*(targer[i]-tmp[i]);
    //    / printf("Error %f diferencia %f semantica %f - targer %f \n", RMSE, (tmp[i]-targer[i])*(tmp[i]-targer[i]) ,tmp[i], targer[i]);
    }
    
    t = sqrt(RMSE/nrow);
    printf("error %f \n", t);
}

__global__ void poblacion(float *p, int size){
    const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
    for (int i=0; i<size; i++) {
        printf(" %f ", p[tid*size+i]);
    }
}

__global__ void dataIn(float *p, int size){
    const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
    for (int i=0; i<size; i++) {
        printf(" %f ", p[tid*size+i]);
    }
}

int main(int argc, char **argv){
    
    srand(time(NULL)); /*!< Initialization of the seed for the generation of random numbers*/

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
    std::string la (pathTrace);

    std::string outputNameFiles(output_model); ///*!< Name of file for save the output files*/

    printf("\n Starting GsgpCuda \n\n");

    readConfigFile(&config); /*!< reading the parameters of the algorithm */

    countInputFile(trainFile, nrow, nvar);

    countInputFile(testFile, nrowTest, nvar);

    nvar--;

    const int individualLength = config.maxIndividualLength; /*!< Variable that stores maximum depth for individuals */
    
    int sizeMemPopulation = sizeof(float) * config.populationSize * individualLength; /*!< Variable that stores size in bytes for initial population*/
    
    int twoSizeMemPopulation = sizeof(float) * (config.populationSize*2); /*!< Variable that stores twice the size in bytes of an initial population to store random numbers*/
    
    int sizeMemIndividuals = sizeof(float) * config.populationSize; /*!< Variable that stores size in bytes of the number of individuals in the initial population*/
    
    long int twoSizePopulation = (config.populationSize*2); /*!< Variable storing twice the initial population of individuals to generate random positions*/
    
    long int sizeMemSemanticTrain = sizeof(float)*(config.populationSize*nrow); /*!< Variable that stores the size in bytes of semantics for the entire population with training data*/
    
    long int sizeMemSemanticTest = sizeof(float)*(config.populationSize*nrowTest); /*!< Variable that stores the size in bytes of semantics for the entire population with test data*/
    
    long int sizeMemDataTrain = sizeof(float)*(nrow*nvar); /*!< Variable that stores the size in bytes the size of the training data*/
    
    long int sizeMemDataTest = sizeof(float)*(nrowTest*nvar); /*!< Variable that stores the size in bytes the size of the test data*/
    
    long int sizeElementsSemanticTrain = (config.populationSize*nrow); /*!< Variable that stores training data elements*/
    
    long int sizeElementsSemanticTest = (config.populationSize*nrowTest); /*!< Variable that stores test data elements*/
    
    long int vectorTracesMem = (sizeof(entry_)*config.numberGenerations*config.populationSize); /*!< Variable that stores the size in bytes of the structure to store the survival record*/

    int gridSize,minGridSize,blockSize; /*!< Variables that store the execution configuration for a kernel in the GPU*/
    
    int gridSizeTest,minGridSizeTest,blockSizeTest; /*!< Variables that store the execution configuration for a kernel in the GPU*/
    
    std::string logPath (config.logPath); /* Path of directory for data files and log files generated in execution */

    std::string namePopulation = "_initialPopulation"; /*!< Name of file for save the initial population  */

    namePopulation = outputNameFiles + namePopulation;

    std::string nameRandomTrees = "_randomTrees"; /*!< name of file for save the random trees */
    
    nameRandomTrees = outputNameFiles + nameRandomTrees;
    
    if (!la.empty()) {
        countInputFile(path_test, nrowTest, nvar);
        nvar--;
 
        namePopulation = la + namePopulation;
        nameRandomTrees = la + nameRandomTrees;

        std::string outFile (pathOutFile);
        outFile = logPath + outFile;

        int sizeDataTest = sizeof(float)*(nrowTest*nvar); /*!< Variable that stores the size in bytes the size of the test data*/

        int sizeDataTestTarget = sizeof(float)*(nrowTest); /*!< Variable that stores the size in bytes the size of the target data */

        float *unssenDataTest, *dUnssenDataTest ,*unssenDataTestTarget, *hsalidas, *dsalidas; /*!< This vector pointers to store the individuals of the test data and target data */

        unssenDataTest = (float *)malloc(sizeDataTest); /*!< Reserve memory on host*/

        unssenDataTestTarget = (float *)malloc(sizeDataTestTarget); /*!< Reserve memory on host*/

        checkCudaErrors(cudaMalloc((void **)&dUnssenDataTest, sizeDataTest));

        hsalidas = (float *)malloc(sizeDataTestTarget); /*!< Reserve memory on host*/

        checkCudaErrors(cudaMalloc((void **)&dsalidas, sizeDataTest));

        readInpuTestData(path_test, unssenDataTest, unssenDataTestTarget, nrowTest, nvar);

        cudaMemcpy(dUnssenDataTest, unssenDataTest, sizeDataTest, cudaMemcpyHostToDevice); 

        //dataIn<<<1,1>>>(dUnssenDataTest, nrowTest);
        
        float *initPopulation, *randomTress, *dInitialPopulation,*dRandomTrees; /*!< This vector pointers to store the individuals of the initial population and random trees */

        initPopulation = (float*)malloc(sizeMemPopulation); /*!<  Variable that stores the size in bytes the initial population */

        randomTress = (float*)malloc(sizeMemPopulation);  /*!< Variable that stores the size in bytes the initial population */

        checkCudaErrors(cudaMalloc((void **)&dRandomTrees, sizeMemPopulation)); 

        checkCudaErrors(cudaMalloc((void **)&dInitialPopulation, sizeMemPopulation));

        long int sizeMemSemanticTest = sizeof(float)*(config.populationSize*nrowTest); /*!< Variable that stores the size in bytes of semantics for the entire population with test data*/

        float *uStackInd; /*!< auxiliary pointer vectors for the interpreter and calculate the semantics for the populations and assignment in the GPU*/
        int   *uPushGenes;
        checkCudaErrors(cudaMalloc((void**)&uPushGenes, sizeMemIndividuals));
        checkCudaErrors(cudaMalloc((void**)&uStackInd, sizeMemPopulation));  

        float *uSemanticCases, *hSemanticCases, *uSemanticRandomTrees,*hSemanticRandomTrees; /*!< pointer of vectors that contain the semantics of an individual in the population, calculated with the training set and test in generation g and its allocation in GPU*/
        checkCudaErrors(cudaMalloc((void**)&uSemanticCases,sizeMemSemanticTest));            
        checkCudaErrors(cudaMalloc((void**)&uSemanticRandomTrees,sizeMemSemanticTest));   
        hSemanticCases = (float*)malloc(sizeMemSemanticTest);
        hSemanticRandomTrees= (float*)malloc(sizeMemSemanticTest);             


        readPopulation(initPopulation, randomTress, config.populationSize, individualLength, logPath, namePopulation, nameRandomTrees);

        ///*!<return the initial population of the device to the host*/
        cudaMemcpy(dInitialPopulation, initPopulation, sizeMemPopulation, cudaMemcpyHostToDevice); 
        cudaMemcpy(dRandomTrees, randomTress, sizeMemPopulation, cudaMemcpyHostToDevice); 
        
        ///poblacion<<<1,1>>>(dInitialPopulation,individualLength);

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeSemantics, 0, config.populationSize); /*!< heuristic function used to choose a good block size is to aim at high occupancy*/
        gridSize = (config.populationSize + blockSize - 1) / blockSize; /*!< round up according to array size*/            
        printf("gird %i and blovk %i \n", gridSize, blockSize);
        /*!< invokes the GPU to interpret the initial population with data train*/
        computeSemantics<<< gridSize, blockSize >>>(dInitialPopulation, uSemanticCases, individualLength, dUnssenDataTest, nrowTest, nvar, uPushGenes, uStackInd);
        cudaMemcpy(hSemanticCases,uSemanticCases, sizeMemSemanticTest,cudaMemcpyDeviceToHost);

        computeSemantics<<< gridSize, blockSize >>>(dRandomTrees, uSemanticRandomTrees, individualLength, dUnssenDataTest, nrowTest, nvar, uPushGenes, uStackInd);
        cudaMemcpy(hSemanticRandomTrees,uSemanticRandomTrees, sizeMemSemanticTest,cudaMemcpyDeviceToHost);

        /*!< Create file for saved results of best model with the unseen data*/
        std::ofstream OUT(outFile,ios::out);

        /*!< Create file for saved results of best model with the unseen data*/
        std::ofstream OUTSem("semantica.txt",ios::out);
        for (int i=0; i<config.populationSize; i++) {
            for (int j=0; j<nrowTest; j++) {
                OUTSem<<hSemanticCases[i*nrowTest+j]<<" ";
            }
            OUTSem<<endl;
        }
        std::ofstream OUTSemRt("semanticaRT.txt",ios::out);
        for (int i=0; i<config.populationSize; i++) {
            for (int j=0; j<nrowTest; j++) {
                OUTSemRt<<hSemanticRandomTrees[i*nrowTest+j]<< " ";
            }
            OUTSemRt<<endl;
        }
        evaluate_data(pathTrace, config.numberGenerations, hSemanticCases, hSemanticRandomTrees, OUT, config.logPath, nrowTest, config.populationSize, nvar,hsalidas);
        
        printf("Validacion \n");
        validacion(outFile, unssenDataTestTarget,  nrowTest);
       
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
    }else {
        
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
 
        float *dInitialPopulation,*dRandomTrees,*hInitialPopulation,*hRandomTrees;  /*!< This block contains the vectors of pointers to store the population and random trees and space allocation in the GPU*/
        hInitialPopulation = (float *)malloc(sizeMemPopulation); 
        hRandomTrees = (float *)malloc(sizeMemPopulation); 
        checkCudaErrors(cudaMalloc((void **)&dRandomTrees, sizeMemPopulation)); 
        checkCudaErrors(cudaMalloc((void **)&dInitialPopulation, sizeMemPopulation));

        entry  *vectorTraces; /*!< This block contains the vectors of pointers to store the structure to keep track of mutation and survival and space allocation in the GPU*/
        checkCudaErrors(cudaMallocManaged(&vectorTraces,vectorTracesMem));
        
        float *uDataTrain, *uDataTest, *uDataTrainTarget, *uDataTestTarget;  /*!< this block contains the pointer of vectors for the input data and target values ​​and assignment in the GPU*/
        checkCudaErrors(cudaMallocManaged(&uDataTrain, sizeMemDataTrain));
        checkCudaErrors(cudaMallocManaged(&uDataTest, sizeMemDataTest));      
        checkCudaErrors(cudaMallocManaged(&uDataTrainTarget, sizeof(float)*nrow));
        checkCudaErrors(cudaMallocManaged(&uDataTestTarget, sizeof(float)*nrowTest));            

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
        float *tempSemantic,*tempFitnes,*tempSemanticTest,*tempFitnesTest; /*!< temporal Variables to perform the movement of pointers in survival*/

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
        
        int result,incx1=1,indexBestIndividual; /*!< this section makes use of the isamin de cublas function to determine the position of the best individual*/
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
        init<<<gridSize, blockSize>>>(time(NULL), State); /*!< initializes the random number generator*/
        cudaErrorCheck("init");     

        float *indexRandomTrees; /*!< vector of pointers to save random positions of random trees and allocation in GPU*/
        checkCudaErrors(cudaMallocManaged(&indexRandomTrees,twoSizeMemPopulation));         
        /*!< main GSGP cycle*/
        for ( int generation=1; generation<=config.numberGenerations; generation++){

            /*!< register execution time*/
            cudaEventRecord(startGsgp);
            gridSize =0, blockSize=0;
            /*!< invokes the GPU to initialize the random positions of the random trees*/
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initializeIndexRandomTrees, 0, twoSizePopulation);
            gridSize = (twoSizePopulation + blockSize - 1) / blockSize;
            //printf("grid %i blocksize %i \n", gridSize, blockSize);
            initializeIndexRandomTrees<<<gridSize,blockSize >>>( config.populationSize, indexRandomTrees, State );
            cudaErrorCheck("initializeIndexRandomTrees");

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, geometricSemanticMutation, 0, sizeElementsSemanticTrain); 
            gridSize = (sizeElementsSemanticTrain + blockSize - 1) / blockSize;
            /*!< geometric semantic mutation with semantic train*/
            geometricSemanticMutation<<< gridSize, blockSize >>>(uSemanticTrainCases, uSemanticRandomTrees,uSemanticTrainCasesNew,
            config.populationSize, nrow, sizeElementsSemanticTrain, generation, indexRandomTrees, vectorTraces);
            cudaErrorCheck("geometricSemanticMutation");
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeError, 0, config.populationSize); 

            gridSize = (config.populationSize + blockSize - 1) / blockSize;
            /*!< invokes the GPU to calculate the error (RMSE) the new population*/
            computeError<<< gridSize,blockSize >>>(uSemanticTrainCasesNew, uDataTrainTarget, uFitNew, nrow);
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
            config.populationSize, nrowTest, sizeElementsSemanticTest, generation, indexRandomTrees, vectorTraces);
            cudaErrorCheck("geometricSemanticMutation");

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeTest, computeError, 0, config.populationSize); 
            gridSizeTest = (config.populationSize + blockSizeTest - 1) / blockSizeTest;
            /*!< invokes the GPU to calculate the error (RMSE) the new population*/
            computeError<<< gridSizeTest,blockSizeTest >>>(uSemanticTestCasesNew, uDataTestTarget, uFitTestNew, nrowTest);
            cudaErrorCheck("computeError");
         
            /*!< set byte values*/
            cudaMemset(indexRandomTrees,0,twoSizeMemPopulation);
            cudaDeviceSynchronize();
         
            /*!< this section performs survival by updating the semantic and fitness vectors respectively*/
            int index = generation-1;
            int tmpIndex = 0;
            if(uFitNew[indexBestOffspring] > uFit[indexBestIndividual]){
                //printf("Fue mejor el padre por lo tanto sucede una supervivencia en la generacion %i el indice del peor de los hijos es %i el mejor de los padres %i \n", generation, indexWorstOffspring, indexBestIndividual);
                for (int i = 0; i < nrow; ++i){
                    uSemanticTrainCasesNew[indexWorstOffspring*nrow+i] = uSemanticTrainCases[indexBestIndividual*nrow+i];
                }

                uFitNew[indexWorstOffspring] = uFit[indexBestIndividual];
                tmpIndex = indexBestIndividual;
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
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].firstParent = tmpIndex;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].secondParent = indexWorstOffspring;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].number=tmpIndex;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].event = -1;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].newIndividual = tmpIndex;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].mark=1;
                vectorTraces[(index*config.populationSize)+indexWorstOffspring].mutStep = 0;
                
                tempFitnesTest = uFitTest;
                uFitTest = uFitTestNew;
                uFitTestNew = tempFitnesTest;
                tempSemanticTest = uSemanticTestCases;
                uSemanticTestCases = uSemanticTestCasesNew;
                uSemanticTestCasesNew = tempSemanticTest;
                indexBestIndividual = indexWorstOffspring;
            }else{
                //printf("Fue mejor el hijo por lo tanto sucede una Mutacion en la generacion %i, el mejor de los padres %i el mejor de los hijos %i \n", generation, indexBestIndividual, indexBestOffspring);
                vectorTraces[(index*config.populationSize)+indexBestOffspring].firstParent = vectorTraces[(index*config.populationSize)+indexBestOffspring].firstParent;
                vectorTraces[(index*config.populationSize)+indexBestOffspring].secondParent =  vectorTraces[(index*config.populationSize)+indexBestOffspring].secondParent;
                vectorTraces[(index*config.populationSize)+indexBestOffspring].number= vectorTraces[(index*config.populationSize)+indexBestOffspring].number;
                vectorTraces[(index*config.populationSize)+indexBestOffspring].event =  vectorTraces[(index*config.populationSize)+indexBestOffspring].event;
                vectorTraces[(index*config.populationSize)+indexBestOffspring].newIndividual =  vectorTraces[(index*config.populationSize)+indexBestOffspring].newIndividual;
                vectorTraces[(index*config.populationSize)+indexBestOffspring].mark= vectorTraces[(index*config.populationSize)+indexBestOffspring].mark=1;
                vectorTraces[(index*config.populationSize)+indexBestOffspring].mutStep =  vectorTraces[(index*config.populationSize)+indexBestOffspring].mutStep;
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
        //markTracesGeneration(vectorTraces, config.populationSize, config.numberGenerations,  indexBestIndividual);
        saveTraceComplete(logPath, vectorTraces, config.numberGenerations, config.populationSize);
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
        /*!< all device allocations are removed*/
        cudaDeviceReset();
        } 
    return 0;
}

