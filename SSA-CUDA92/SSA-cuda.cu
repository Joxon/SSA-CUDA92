#include "SSA-cuda.h"

__global__ void getSquareKernel(float *dSqrt, float *dData, size_t dataSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dataSize)
    {
        dSqrt[i] = dData[i] * dData[i];
    }
}

__global__ void getSumKernel(float * dSum, float * dData, size_t dataSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dataSize)
    {
        atomicAdd(dSum, dData[i]);
    }
}

//__global__ void init_position_kernel(float *solution_dev)
//{
//    int tid = threadIdx.x;
//    //rand() is illegal here
//    solution_dev[tid]= ((float)rand() / (RAND_MAX + 1.0f)) * 200.0f - 100.0f;
//}

//__global__ void choose_vibration_kernel(int *max_index_dev,
//                                        float *max_intensity_dev,
//                                        Vibration targetVibr,
//                                        const std::vector<Vibration>& vibrations,
//                                        const std::vector<float>& distances,
//                                        float attenuation_factor)
//{
//    int tid = threadIdx.x;
//    if(vibrations[tid].position == targetVibr.position) return;
//
//    float intensity = vibrations[tid].intensity* exp(-distances[tid] / attenuation_factor);
//    if(intensity > *max_intensity_dev)
//    {
//        *max_index_dev = tid;
//        *max_intensity_dev = intensity;
//    }
//}

__global__ void curandInitKernel(unsigned long long seed,
                                 curandState * devStates,
                                 size_t devStateSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < devStateSize)
    {
        curand_init(seed, i, 0, &devStates[i]);
    }
}

__global__ void randomWalkKernel(float ** devVibPosSol,
                                 float * devPrevMove,
                                 float * devPosSol,
                                 unsigned * devDimMask,
                                 float * devTgtVibPosSol,
                                 curandState * devStates,
                                 size_t popSize,
                                 size_t probDim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < probDim)
    {
        curandState localState = devStates[i];

        //curand_uniform returns 0.0 - 1.0
        devPrevMove[i] *= curand_uniform(&devStates[i]);

        float targetPosition = devDimMask[i] ?
            devVibPosSol[curand(&devStates[i]) % popSize][i] :
            devTgtVibPosSol[i];

        devPrevMove[i] += curand_uniform(&devStates[i]) * (targetPosition - devPosSol[i]);

        devPosSol[i] += devPrevMove[i];

        devStates[i] = localState;
    }
}

__global__ void getAllStdDev(float * dStdDevArray, float *dData, size_t probDim, size_t popSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < probDim)
    {
        //float mean = thrust::reduce(thrust::device,
        //                            dData + popSize * i,
        //                            dData + popSize * (i + 1)) / popSize;

        float sum = 0.0f;
        int base = popSize * i;
        for (int j = 0; j < popSize; ++j)
        {
            sum += dData[base + j];
        }
        float mean = sum / popSize;

        //float variance = thrust::transform_reduce(thrust::device,
        //                                          dData + popSize * i,
        //                                          dData + popSize * (i + 1),
        //                                          variance_functor<float>(mean),
        //                                          0.0f,
        //                                          thrust::plus<float>()) / popSize;
        sum = 0.0f;
        for (int j = 0; j < popSize; ++j)
        {
            float x = dData[base + j] - mean;
            sum += x * x;
        }
        float variance = sum / popSize;

        dStdDevArray[i] = sqrtf(variance);

        //cudaFree(dData);
    }
}

//__global__ void vibrationGenerationFillDataKernel(Spider * dPop, float * dData, unsigned popSize, size_t solIdx)
//{
//    int i = threadIdx.x;
//
//    if(i < popSize)
//    {
//        dData[i] = dPop[i].position.solution[solIdx];
//    }
//}

__global__ void chooseVibrationKernel(int * dMaxIndex,
                                      float * dMaxIntensity,
                                      float ** dVibPosSol,
                                      float ** dSpdTgtPosSol,
                                      float * dVibInt,
                                      float ** dDist,
                                      size_t popSize,
                                      size_t probDim,
                                      float attenuation_factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < popSize)
    {
        for (int j = 0; j < popSize; ++j)
        {
            unsigned isTargetVibr = 1;
            for (int k = 0; k < probDim; ++k)
            {
                if (dVibPosSol[j][k] != dSpdTgtPosSol[i][k])
                {
                    isTargetVibr = 0;
                    break;
                }
            }
            if (isTargetVibr == 1)
            {
                continue;
            }

            float intensity = dVibInt[j] * exp((-dDist[i][j]) / attenuation_factor);
            if (intensity > dMaxIntensity[i])
            {
                dMaxIndex[i] = j;
                dMaxIntensity[i] = intensity;
            }
        }
    }
}

__global__ void getAllFitnesses(float * dFitness,
                                float * dPopPosSol,
                                size_t popSize,
                                size_t probDim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < popSize)
    {
        /*dFitness[i] = thrust::transform_reduce(thrust::device,
                                               dPopPosSol + i * probDim,
                                               dPopPosSol + (i + 1) * probDim,
                                               square_functor<float>(),
                                               0.0,
                                               thrust::plus<float>());
                                               */
        float sum = 0.0;

        for (int j = 0; j < probDim; ++j)
        {
            float k = dPopPosSol[i * popSize + j];
            sum += k * k;
        }

        dFitness[i] = sum;
    }
}

__global__ void getAllDistances(float * dDist1D,
                                float ** dPopPosSol2,
                                size_t popSize,
                                size_t probDim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < popSize)
    {
        int base = ((2 * popSize - i - 1)*i) / 2;

        for (int j = i + 1; j < popSize; ++j)
        {
            float dist = 0.0;
            for (int k = 0; k < probDim; ++k)
            {
                dist += fabs(dPopPosSol2[i][k] - dPopPosSol2[j][k]);
            }

            int offset = (j - i - 1);
            dDist1D[base + offset] = dist;
        }
    }
}