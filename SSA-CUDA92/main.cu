#include "SSA.h"
#include "SSA-cuda.h"

class MyProblem : public Problem
{
public:
    MyProblem(unsigned int dimension) : Problem(dimension) {}

#if defined GET_FITNESS_THRUST
    float getFitnessThrust(const std::vector<float>& solution)
    {
        //thrust::device_vector<float> dSol(solution);
        return thrust::transform_reduce(solution.begin(),
                                        solution.end(),
                                        square_functor<float>(),
                                        0.0,
                                        thrust::plus<float>());

        //
        // thrust version
        //

        // copy
        //thrust::device_vector<float> dSol(solution);

        // debug
        //thrust::host_vector<float> hSol(dSol);
        //size_t solSize = solution.size();
        //for(size_t i = 0; i < solSize; ++i)
        //{
        //    printf("%.1f ", hSol[i]);
        //}
        //printf("\n");

        // square
        //using namespace thrust::placeholders;
        //thrust::transform(dSol.begin(), dSol.end(),
        //                  dSol.begin(),
        //                  _1*_1);

        // debug
        //hSol = dSol;
        //for(size_t i = 0; i < solSize; ++i)
        //{
        //    printf("%.1f ", hSol[i]);
        //}
        //printf("\n");

        // get sum
        //float sum = thrust::reduce(dSol.begin(), dSol.end());

        //thrust::device_vector<float> dSol(solution);
        //thrust::device_vector<float> dSolB(solution);
        //thrust::device_vector<float> dSolSquared(solution.size());
        //thrust::transform(dSol.begin(), dSol.end(),
        //                  dSolB.begin(),
        //                  dSolSquared.begin(),
        //                  thrust::multiplies<float>());

        //return thrust::reduce(dSolSquared.begin(), dSolSquared.end());
    }

#elif defined GET_FITNESS_CUDA
    float getFitnessCuda(const std::vector<float>& solution)
    {
        //
        // kernel version
        //

        size_t solSize = solution.size();

        float *dSol;
        checkCudaErrors(cudaMalloc<float>(&dSol, solSize * sizeof(float)));
        checkCudaErrors(cudaMemcpy(dSol, &solution[0], solSize * sizeof(float), cudaMemcpyHostToDevice));

        float *dSqrt;
        checkCudaErrors(cudaMalloc<float>(&dSqrt, solSize * sizeof(float)));

        getSquareKernel KERNEL_ARGS2(DIM_BLOCKS(solSize), DIM_THREADS) (dSqrt, dSol, solSize);

        float sum = 0.0;
        float *dSum;
        checkCudaErrors(cudaMalloc<float>(&dSum, sizeof(float)));
        checkCudaErrors(cudaMemcpy(dSum, &sum, sizeof(float), cudaMemcpyHostToDevice));
        getSumKernel KERNEL_ARGS2(DIM_BLOCKS(solSize), DIM_THREADS) (dSum, dSqrt, solSize);

        checkCudaErrors(cudaMemcpy(&sum, dSum, sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(dSqrt));
        checkCudaErrors(cudaFree(dSum));

        return sum;
    }

#elif defined GET_FITNESS_HOST
    float getFitnessHost(const std::vector<float>& solution)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < solution.size(); ++i)
        {
            sum += solution[i] * solution[i];
        }
        return sum;
    }

#endif // GET_FITNESS_THRUST

};

int main(int argc, char **argv)
{
    unsigned problemDimension = 30;
    unsigned populationSize = 30;
    SSA ssa(new MyProblem(problemDimension), populationSize);

    int iterationTimes = 1000;
    float attenuationRate = 1.0f;
    float pchange = 0.7f;
    float pmask = 0.1f;
    ssa.run(iterationTimes, attenuationRate, pchange, pmask);

    getchar();
    return 0;
}