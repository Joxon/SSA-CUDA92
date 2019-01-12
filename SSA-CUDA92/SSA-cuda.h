#pragma once

#include <device_functions.h>
#include <device_launch_parameters.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// thrust::device_vector can not be used in __global__ kernel
#include <thrust\execution_policy.h>
#include <thrust\device_vector.h>
#include <thrust\transform_reduce.h>
#include <thrust\functional.h>
#include <thrust\copy.h>
#include <thrust\extrema.h>

#include "helper_cuda.h"

#include "SSA.h"

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif
// Now launch your kernel using the appropriate macro:
//kernel KERNEL_ARGS2(dim3(nBlockCount), dim3(nThreadCount)) (param1);

#define MAX_THREADS_PER_MP 2048
#define MAX_THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK 64
#define DIM_THREADS (dim3 (THREADS_PER_BLOCK, 1))
#define DIM_BLOCKS(probSize) (dim3 (((probSize / THREADS_PER_BLOCK) + 1), 1))

//#define GET_FITNESS_THRUST
//#define GET_FITNESS_CUDA
//#define GET_FITNESS_HOST

template <typename T>
struct square_functor
{
    __host__ __device__
        T operator()(const T& x) const
    {
        return x * x;
    }
};

template <typename T>
struct variance_functor
{
    const T mean;

    __host__ __device__
        variance_functor(T m) : mean(m) {}

    __host__ __device__
        T operator()(const T& data) const
    {
        float x = (data - mean);
        return x * x;
    }
};

__global__ void getSquareKernel(float *dSqrt,
                                float *dData,
                                size_t dataSize);

__global__ void getSumKernel(float *dSum,
                             float *dData,
                             size_t dataSize);

__global__ void curandInitKernel(unsigned long long seed,
                                 curandState *devState,
                                 size_t devStateSize);

__global__ void randomWalkKernel(float ** devVibPosSol,
                                 float * devPrevMove,
                                 float * devPosSol,
                                 unsigned * devDimMask,
                                 float * devTgtVibPosSol,
                                 curandState * devStates,
                                 size_t popSize,
                                 size_t probDim);

__global__ void getAllStdDev(float *dSum,
                             float *dData,
                             size_t probDim,
                             size_t popSize);

//__global__ void vibrationGenerationFillDataKernel(Spider *dPop,
//                                                  float *dData,
//                                                  unsigned popSize,
//                                                  size_t solIdx);

__global__ void chooseVibrationKernel(int *dMaxIndex,
                                      float *dMaxIntensity,
                                      float ** dVibPosSol,
                                      float ** dSpdTgtPosSol,
                                      float * dVibInt,
                                      float ** dDist,
                                      size_t popSize,
                                      size_t probDim,
                                      float attenuation_factor);

__global__ void getAllFitnesses(float * dFitness,
                                float * dPopPosSol,
                                size_t popSize,
                                size_t probDim);

__global__ void getAllDistances(float * dDist1D,
                                float ** dPopPosSol2,
                                size_t popSize,
                                size_t probDim);
