#include "SSA.h"
#include "SSA-cuda.h"

Position Position::initPosition(Problem* problem)
{
    std::vector<float> solution(problem->dimension);

    for (unsigned i = 0; i < problem->dimension; ++i)
    {
        solution[i] = randu() * 200.0f - 100.0f;
    }

    return Position(solution, FLT_MAX);
}

Vibration::Vibration(const Position& position) :
    intensity(fitnessToIntensity(position.fitness)), position(position)
{}

Vibration::Vibration(float intensity, const Position& position) :
    intensity(intensity), position(position)
{}

Spider::Spider(const Position& position) :
    position(position), inactiveDegree(0)
{
    targetVibr = Vibration(0, position);
    dimensionMask.resize(position.solution.size());
    previousMove.resize(position.solution.size());
}

SSA::SSA(Problem* problem, unsigned int popSize) :
    problem(problem), dimension(problem->dimension)
{
    srand((unsigned int)(time(NULL) + (int)this));

    population.reserve(popSize);
    vibrations.reserve(popSize);
    distances.resize(popSize);
    for (unsigned i = 0; i < popSize; ++i)
    {
        Position position = Position::initPosition(problem);
        population.push_back(Spider(position));
        distances[i].resize(popSize);
    }

    globalBestPosition = population[0].position;
}

float Vibration::intensityAttenuation(float attenuation_factor, float distance) const
{
    return intensity * exp(-distance / attenuation_factor);
}

float Vibration::fitnessToIntensity(float fitness)
{
    return  log(1.0f / (fitness - C) + 1.0f);
}

//double Vibration::C = -1E-100;
float Vibration::C = -FLT_MIN;

void SSA::printHeader()
{
    std::cout << "               SSA starts at " << getTimeStdString() << std::endl
        << "==============================================================" << std::endl
        << " iter    optimum    pop_min  base_dist  mean_dist time_elapsed" << std::endl
        << "==============================================================" << std::endl;
}

void SSA::printContent()
{
    auto current_time = std::chrono::high_resolution_clock::now();
    printf("%5d %.3e %.3e %.3e %.3e ",
           iteration,
           globalBestPosition.fitness,
           populationBestFitness,
           attenuationBase,
           meanDistance);
    std::cout << getTimeStdString(std::chrono::duration_cast<std::chrono::milliseconds>(current_time - startTime)) << std::endl;
}

void SSA::printFooter()
{
    std::cout << "==============================================================" << std::endl;
    printContent();
    std::cout << "==============================================================" << std::endl;
}

std::string getTimeStdString()
{
    using namespace std::chrono;

    auto time_t = system_clock::to_time_t(system_clock::now());
    auto ttm = localtime(&time_t);

    char time_str[80];
    strftime(time_str, 80, "%Y-%m-%d %H:%M:%S", ttm);
    return std::string(time_str);
}

std::string getTimeStdString(std::chrono::milliseconds ms)
{
    long long msc = ms.count();
    char time_str[80];
    sprintf(time_str, "%02d:%02d:%02d.%03d",
            int(msc / 3600000),
            int(msc / 60000 % 60),
            int(msc / 1000 % 60),
            int(msc % 1000));
    return std::string(time_str);
}

void SSA::run(int maxIteration,
              float attenuationRate,
              float pChange,
              float pMask)
{
    printHeader();
    startTime = std::chrono::high_resolution_clock::now();

    checkCudaErrors(cudaMalloc<curandState>(&dStates, dimension * sizeof(curandState)));
    curandInitKernel KERNEL_ARGS2(DIM_BLOCKS(dimension), DIM_THREADS)
        (rand(), dStates, dimension);

    size_t popSize = population.size();
    for (iteration = 1; iteration <= maxIteration; ++iteration)
    {
        //3
        fitnessCalculation();
        //1
        vibrationGeneration(attenuationRate);

        for (int i = 0; i < popSize; ++i)
        {
            population[i].maskChanging(pChange, pMask);
            //2
            population[i].randomWalk(vibrations);
        }

        if ((iteration == 1)
            || (iteration == 10)
            || (iteration < 1001 && iteration % 100 == 0)
            || (iteration < 10001 && iteration % 1000 == 0)
            || (iteration < 100001 && iteration % 10000 == 0))
        {
            printContent();
        }
    }

    checkCudaErrors(cudaFree(dStates));

    --iteration;
    printFooter();
}

//3
void SSA::fitnessCalculation()
{
    populationBestFitness = FLT_MAX;

    size_t popSize = population.size();
    size_t probDim = dimension;

    /*
    * getAllFitnesses BEGINS
    */

    //fitness: w/o
    float* dFitness;
    checkCudaErrors(cudaMalloc<float>(&dFitness, popSize * sizeof(float)));

    //population[i].position.solution: r/o
    float *hPopPosSol;
    hPopPosSol = (float*)malloc(popSize * probDim * sizeof(float));
    for (unsigned i = 0; i < popSize; ++i)
    {
        memcpy(hPopPosSol + i * probDim,
               &population[i].position.solution[0],
               probDim * sizeof(float));
    }
    float *dPopPosSol;
    checkCudaErrors(cudaMalloc<float>(&dPopPosSol, popSize * probDim * sizeof(float)));
    checkCudaErrors(cudaMemcpy(dPopPosSol, hPopPosSol, popSize * probDim * sizeof(float), cudaMemcpyHostToDevice));
    free(hPopPosSol);

    //population[i].position.fitness: w/o
    //float *hPopPosFit;
    //hPopPosSol = (float*)malloc(popSize * sizeof(float));
    //for(unsigned i = 0; i < popSize; ++i)
    //{
    //    hPopPosSol[i] = population[i].position.fitness;
    //}
    //float *dPopPosFit;
    //checkCudaErrors(cudaMalloc<float>(&dPopPosSol, popSize * sizeof(float)));
    //checkCudaErrors(cudaMemcpy(dPopPosFit, hPopPosFit, popSize * sizeof(float), cudaMemcpyHostToDevice));
    //free(hPopPosFit);

    getAllFitnesses KERNEL_ARGS2(DIM_BLOCKS(popSize), DIM_THREADS)
        (dFitness, dPopPosSol, popSize, probDim);

    float* hFitness;
    hFitness = (float*)malloc(popSize * sizeof(float));
    checkCudaErrors(cudaMemcpy(hFitness, dFitness, popSize * sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < popSize; ++i)
    {
        population[i].position.fitness = hFitness[i];
        if (hFitness[i] < globalBestPosition.fitness)
        {
            globalBestPosition = population[i].position;
        }
        if (hFitness[i] < populationBestFitness)
        {
            populationBestFitness = hFitness[i];
        }
    }
    checkCudaErrors(cudaFree(dFitness));
    checkCudaErrors(cudaFree(dPopPosSol));
    free(hFitness);

    //thrust::device_ptr<float> dFitnessPtr =
    //    thrust::device_pointer_cast<float>(dFitness);
    //thrust::device_vector<float>::iterator hMinFitPtr =
    //    thrust::min_element(dFitnessPtr, dFitnessPtr + popSize);

    //float * hMinFitPtr = thrust::min_element(hFitness, hFitness + popSize);
    //float hMinFit = *hMinFitPtr;
    //int hMinFitIdx = hMinFitPtr - hFitness;
    //if(hMinFit < globalBestPosition.fitness)
    //{
    //    globalBestPosition = population[hMinFitIdx].position;
    //}
    //if(hMinFit < populationBestFitness)
    //{
    //    populationBestFitness = hMinFit;
    //}


    /*
    * getAllDistances BEGINS
    */

    //distances[i][j] - 2D
    //float ** hDist;
    //hDist = (float**)malloc(popSize * sizeof(float*));
    //float ** dDist;
    //checkCudaErrors(cudaMalloc<float*>(&dDist, popSize * sizeof(float*)));
    //float * dDistData;
    //checkCudaErrors(cudaMalloc<float>(&dDistData, popSize * popSize * sizeof(float)));
    //for(size_t i = 0; i < popSize; ++i)
    //{
    //    hDist[i] = dDistData + i * popSize;
    //}
    //checkCudaErrors(cudaMemcpy(dDist, hDist, popSize * sizeof(float*), cudaMemcpyHostToDevice));
    //free(hDist);

    //distances[i][j] - 1D: w/o
    //index = ((2n-i-1)*i)/2 + (j-i-1)
    float *dDist1D;
    size_t dDist1DCount = (popSize * (popSize - 1) / 2);
    size_t dDist1DSize = dDist1DCount * sizeof(float);
    checkCudaErrors(cudaMalloc<float>(&dDist1D, dDist1DSize));

    //population[i].position.solution[k]: r/o
    float ** hPopPosSol2;
    hPopPosSol2 = (float**)malloc(popSize * sizeof(float*));
    float ** dPopPosSol2;
    checkCudaErrors(cudaMalloc<float*>(&dPopPosSol2, popSize * sizeof(float*)));

    float *hPopPosSolData;
    hPopPosSolData = (float*)malloc(popSize * probDim * sizeof(float));
    float *dPopPosSolData;
    checkCudaErrors(cudaMalloc<float>(&dPopPosSolData, popSize * probDim * sizeof(float)));

    for (size_t i = 0; i < popSize; ++i)
    {
        memcpy(hPopPosSolData + i * probDim,
               &population[i].position.solution[0],
               probDim * sizeof(float));
        hPopPosSol2[i] = dPopPosSolData + i * probDim;
    }

    checkCudaErrors(cudaMemcpy(dPopPosSol2, hPopPosSol2, popSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dPopPosSolData, hPopPosSolData, popSize * probDim * sizeof(float), cudaMemcpyHostToDevice));
    free(hPopPosSol2);
    free(hPopPosSolData);

    getAllDistances KERNEL_ARGS2(DIM_BLOCKS(popSize), DIM_THREADS)
        (dDist1D, dPopPosSol2, popSize, probDim);

    float * hDist1D;
    hDist1D = (float *)malloc(dDist1DSize);
    checkCudaErrors(cudaMemcpy(hDist1D, dDist1D, dDist1DSize, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < popSize; ++i)
    {
        size_t base = ((2 * popSize - i - 1)*i) / 2;
        for (size_t j = i + 1; j < popSize; ++j)
        {
            size_t offset = (j - i - 1);
            float dist = hDist1D[base + offset];
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    meanDistance = thrust::reduce(hDist1D,
                                  hDist1D + dDist1DCount) / dDist1DCount;

    free(hDist1D);
    checkCudaErrors(cudaFree(dDist1D));
    checkCudaErrors(cudaFree(dPopPosSol2));
    checkCudaErrors(cudaFree(dPopPosSolData));
}

//1
void SSA::vibrationGeneration(float attenuationRate)
{
    size_t popSize = population.size();
    vibrations.clear();
    for (int i = 0; i < popSize; ++i)
    {
        vibrations.push_back(std::move(Vibration(population[i].position)));
    }

    /*
    * getAllStdDev BEGINS
    */

    //population[j].position.solution[i]: r/o
    //float hSol[dimension][popSize];
    //[sol 0 of all spiders](len=popSize)[sol 1 of all spiders](len=popSize)...
    size_t solSize = dimension * popSize * sizeof(float);
    float *hSol = (float *)malloc(solSize);
    for (unsigned i = 0; i < dimension; ++i)
    {
        unsigned base = i * dimension;
        for (unsigned j = 0; j < popSize; ++j)
        {
            hSol[base + j] = population[j].position.solution[i];

            //DEBUG
            //printf("%.1f ", hSol[base + j]);
        }
        //printf("\n");
    }

    float *dSol;
    checkCudaErrors(cudaMalloc<float>(&dSol, solSize));
    checkCudaErrors(cudaMemcpy(dSol, hSol, solSize, cudaMemcpyHostToDevice));

    //getStdDev: w/o
    float *dStdDev;
    checkCudaErrors(cudaMalloc<float>(&dStdDev, dimension * sizeof(float)));

    getAllStdDev KERNEL_ARGS2(DIM_BLOCKS(dimension), DIM_THREADS)
        (dStdDev, dSol, dimension, popSize);

    //DEBUG
    //float *hStdDev = (float *)malloc(dimension * sizeof(float));
    //checkCudaErrors(cudaMemcpy(hStdDev, dStdDev, dimension * sizeof(float), cudaMemcpyDeviceToHost));
    //for(int j = 0; j < dimension; ++j)
    //{
    //    printf("%.1f ", hStdDev[j]);
    //}

    thrust::device_ptr<float> dPtrStdDev = thrust::device_pointer_cast<float>(dStdDev);

    attenuationBase = thrust::reduce(dPtrStdDev,
                                     dPtrStdDev + dimension) / dimension;
    float attenuationFactor = attenuationBase * attenuationRate;

    free(hSol);
    checkCudaErrors(cudaFree(dSol));
    checkCudaErrors(cudaFree(dStdDev));

    /*
    * chooseVibrationKernel BEGINS
    */

    //max_index: r/w
    int *hMaxIndex;
    hMaxIndex = (int*)malloc(popSize * sizeof(int));
    for (int i = 0; i < popSize; ++i)
    {
        hMaxIndex[i] = -1;
    }
    //memset(hMaxIndex, -1, popSize * sizeof(int));
    int *dMaxIndex;
    checkCudaErrors(cudaMalloc<int>(&dMaxIndex, popSize * sizeof(int)));
    checkCudaErrors(cudaMemcpy(dMaxIndex, hMaxIndex, popSize * sizeof(int), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemset(dMaxIndex, -1, popSize * sizeof(int)));

    //max_intensity: r/w
    float *hMaxIntensity = (float*)malloc(popSize * sizeof(float));
    for (unsigned i = 0; i < popSize; ++i)
    {
        hMaxIntensity[i] = population[i].targetVibr.intensity;
    }
    float *dMaxIntensity;
    checkCudaErrors(cudaMalloc<float>(&dMaxIntensity, popSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(dMaxIntensity, hMaxIntensity, popSize * sizeof(float), cudaMemcpyHostToDevice));

    //vibrations[j].position.solution[k]: r/o
    float ** hVibPosSol;
    hVibPosSol = (float**)malloc(popSize * sizeof(float*));
    float ** dVibPosSol;
    checkCudaErrors(cudaMalloc<float*>(&dVibPosSol, popSize * sizeof(float*)));

    size_t vibPosSolDataSize = popSize * dimension * sizeof(float);
    float * hVibPosSolData;
    hVibPosSolData = (float*)malloc(vibPosSolDataSize);
    float * dVibPosSolData;
    checkCudaErrors(cudaMalloc<float>(&dVibPosSolData, vibPosSolDataSize));

    for (size_t i = 0; i < popSize; ++i)
    {
        memcpy(hVibPosSolData + i * dimension,
               &vibrations[i].position.solution[0],
               dimension * sizeof(float));
        hVibPosSol[i] = dVibPosSolData + i * dimension;
    }

    checkCudaErrors(cudaMemcpy(dVibPosSolData, hVibPosSolData, vibPosSolDataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dVibPosSol, hVibPosSol, popSize * sizeof(float*), cudaMemcpyHostToDevice));
    free(hVibPosSol);
    free(hVibPosSolData);

    //population[i].targetVibr.position.solution[k]: r/o
    float ** hSpdTgtPosSol;
    hSpdTgtPosSol = (float**)malloc(popSize * sizeof(float*));
    float ** dSpdTgtPosSol;
    checkCudaErrors(cudaMalloc<float*>(&dSpdTgtPosSol, popSize * sizeof(float*)));

    float * hPopTgtPosSolData;
    hPopTgtPosSolData = (float*)malloc(vibPosSolDataSize);
    float * dPopTgtPosSolData;
    checkCudaErrors(cudaMalloc<float>(&dPopTgtPosSolData, vibPosSolDataSize));

    for (size_t i = 0; i < popSize; ++i)
    {
        memcpy(hPopTgtPosSolData + i * dimension,
               &population[i].targetVibr.position.solution[0],
               dimension * sizeof(float));
        hSpdTgtPosSol[i] = dPopTgtPosSolData + i * dimension;
    }

    checkCudaErrors(cudaMemcpy(dPopTgtPosSolData, hPopTgtPosSolData, vibPosSolDataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dSpdTgtPosSol, hSpdTgtPosSol, popSize * sizeof(float*), cudaMemcpyHostToDevice));
    free(hSpdTgtPosSol);
    free(hPopTgtPosSolData);

    //vibrations[j].intensity: r/o
    float * hVibInt = (float*)malloc(popSize * sizeof(float));
    for (size_t i = 0; i < popSize; ++i)
    {
        hVibInt[i] = vibrations[i].intensity;
    }
    float * dVibInt;
    checkCudaErrors(cudaMalloc<float>(&dVibInt, popSize * sizeof(float)));
    checkCudaErrors(cudaMemcpy(dVibInt, hVibInt, popSize * sizeof(float), cudaMemcpyHostToDevice));
    free(hVibInt);

    //distances[i][j]: r/o
    float ** hDist;
    hDist = (float**)malloc(popSize * sizeof(float*));
    float ** dDist;
    checkCudaErrors(cudaMalloc<float*>(&dDist, popSize * sizeof(float*)));

    size_t distSize = popSize * popSize * sizeof(float);
    float * hDistData;
    hDistData = (float*)malloc(distSize);
    float * dDistData;
    checkCudaErrors(cudaMalloc<float>(&dDistData, distSize));

    for (size_t i = 0; i < popSize; ++i)
    {
        memcpy(hDistData + i * popSize,
               &distances[i][0],
               popSize * sizeof(float));
        hDist[i] = dDistData + i * popSize;
    }

    checkCudaErrors(cudaMemcpy(dDistData, hDistData, distSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dDist, hDist, popSize * sizeof(float*), cudaMemcpyHostToDevice));
    free(hDist);
    free(hDistData);

    chooseVibrationKernel KERNEL_ARGS2(DIM_BLOCKS(popSize), DIM_THREADS)(dMaxIndex,
                                                                         dMaxIntensity,
                                                                         dVibPosSol,
                                                                         dSpdTgtPosSol,
                                                                         dVibInt,
                                                                         dDist,
                                                                         popSize,
                                                                         dimension,
                                                                         attenuationFactor);

    checkCudaErrors(cudaMemcpy(hMaxIndex, dMaxIndex, popSize * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hMaxIntensity, dMaxIntensity, popSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Host chooses vibration for each spider
    for (size_t i = 0; i < popSize; ++i)
    {
        Spider *spider = &population[i];

        if (hMaxIndex[i] != -1)
        {
            spider->targetVibr = Vibration(hMaxIntensity[i], vibrations[hMaxIndex[i]].position);
            spider->inactiveDegree = 0;
        }
        else
        {
            ++spider->inactiveDegree;
        }
    }

    checkCudaErrors(cudaFree(dMaxIndex));
    checkCudaErrors(cudaFree(dMaxIntensity));
    checkCudaErrors(cudaFree(dVibPosSol));
    checkCudaErrors(cudaFree(dVibPosSolData));
    checkCudaErrors(cudaFree(dSpdTgtPosSol));
    checkCudaErrors(cudaFree(dPopTgtPosSolData));
    checkCudaErrors(cudaFree(dVibInt));
    checkCudaErrors(cudaFree(dDist));
    checkCudaErrors(cudaFree(dDistData));
}

//2
void Spider::chooseVibrationHost(const std::vector<Vibration>& vibrations,
                                 const std::vector<float>& distances,
                                 float attenuation_factor)
{
    // Choose the max vibration
    int max_index = -1;
    float max_intensity = targetVibr.intensity;

    // Host loop
    for (int i = 0; i < vibrations.size(); ++i)
    {
        if (vibrations[i].position == targetVibr.position)
        {
            continue;
        }
        float intensity = vibrations[i].intensityAttenuation(attenuation_factor, distances[i]);
        if (intensity > max_intensity)
        {
            max_index = i;
            max_intensity = intensity;
        }
    }

    if (max_index != -1)
    {
        targetVibr = Vibration(max_intensity, vibrations[max_index].position);
        inactiveDegree = 0;
    }
    else
    {
        ++inactiveDegree;
    }
}

void Spider::maskChanging(float p_change, float p_mask)
{
    if (randu() > pow(p_change, inactiveDegree))
    {
        inactiveDegree = 0;
        p_mask *= randu();

        for (int i = 0; i < dimensionMask.size(); ++i)
        {
            dimensionMask[i] = (randu()) < p_mask;
        }
    }
}

//2
//THERE MAY BE BUGS
void Spider::randomWalk(const std::vector<Vibration>& vibrations)
{
    /*
    * randomWalkKernel BEGINS
    */

    size_t popSize = vibrations.size();
    size_t probDim = position.solution.size();

    //vibrations[rand() % vibrations.size()].position.solution[i]
    float ** hVibPosSol;
    hVibPosSol = (float**)malloc(popSize * sizeof(float*));
    float ** dVibPosSol;
    checkCudaErrors(cudaMalloc<float*>(&dVibPosSol, popSize * sizeof(float*)));

    float * hVibPosSolData;
    hVibPosSolData = (float*)malloc(popSize * probDim * sizeof(float));
    float * dVibPosSolData;
    checkCudaErrors(cudaMalloc<float>(&dVibPosSolData, popSize * probDim * sizeof(float)));

    for (size_t i = 0; i < popSize; ++i)
    {
        memcpy(hVibPosSolData + i * probDim,
               &vibrations[i].position.solution[0],
               probDim * sizeof(float));
        hVibPosSol[i] = dVibPosSolData + i * probDim;
    }

    checkCudaErrors(cudaMemcpy(dVibPosSolData, hVibPosSolData, popSize * probDim * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dVibPosSol, hVibPosSol, popSize * sizeof(float*), cudaMemcpyHostToDevice));
    free(hVibPosSol);
    free(hVibPosSolData);

    //previousMove[i]: r/w
    float * dPrevMove;
    checkCudaErrors(cudaMalloc<float>(&dPrevMove, probDim * sizeof(float)));
    checkCudaErrors(cudaMemcpy(dPrevMove, &previousMove[0], probDim * sizeof(float), cudaMemcpyHostToDevice));

    //position.solution[i]: r/w
    float * dPosSol;
    checkCudaErrors(cudaMalloc<float>(&dPosSol, probDim * sizeof(float)));
    checkCudaErrors(cudaMemcpy(dPosSol, &position.solution[0], probDim * sizeof(float), cudaMemcpyHostToDevice));

    //dimensionMask[i]: r/o
    //CUDA failed to support bool
    unsigned * hDimMask = (unsigned*)malloc(probDim * sizeof(unsigned));
    for (size_t i = 0; i < probDim; ++i)
    {
        hDimMask[i] = dimensionMask[i] ? 1 : 0;
    }
    unsigned * dDimMask;
    checkCudaErrors(cudaMalloc<unsigned>(&dDimMask, probDim * sizeof(unsigned)));
    checkCudaErrors(cudaMemcpy(dDimMask, hDimMask, probDim * sizeof(unsigned), cudaMemcpyHostToDevice));
    free(hDimMask);

    //targetVibr.position.solution[i]: r/o
    float * dTgtVibPosSol;
    checkCudaErrors(cudaMalloc<float>(&dTgtVibPosSol, probDim * sizeof(float)));
    checkCudaErrors(cudaMemcpy(dTgtVibPosSol, &targetVibr.position.solution[0], probDim * sizeof(float), cudaMemcpyHostToDevice));

    randomWalkKernel KERNEL_ARGS2(DIM_BLOCKS(probDim), DIM_THREADS) (dVibPosSol,
                                                                     dPrevMove,
                                                                     dPosSol,
                                                                     dDimMask,
                                                                     dTgtVibPosSol,
                                                                     dStates,
                                                                     popSize,
                                                                     probDim);

    checkCudaErrors(cudaMemcpy(&previousMove[0], dPrevMove, probDim * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&position.solution[0], dPosSol, probDim * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dVibPosSolData));
    checkCudaErrors(cudaFree(dVibPosSol));
    checkCudaErrors(cudaFree(dPrevMove));
    checkCudaErrors(cudaFree(dPosSol));
    checkCudaErrors(cudaFree(dDimMask));
    checkCudaErrors(cudaFree(dTgtVibPosSol));
}