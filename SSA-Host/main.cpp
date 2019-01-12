#include "SSA.h"

class MyProblem : public Problem
{
public:
    MyProblem(unsigned int dimension) : Problem(dimension) {}

    double getFitness(const std::vector<double>& solution)
    {
        double sum = 0.0;

        for(int i = 0; i < solution.size(); ++i)
        {
            sum += solution[i] * solution[i];
        }

        return sum;
    }
};

int main()
{
    unsigned problemDimension = 30;
    unsigned populationSize = 30;
    SSA ssa(new MyProblem(problemDimension), populationSize);

    int iterationTimes = 10000;
    double attenuationRate = 1.0;
    double pchange = 0.7;
    double pmask = 0.1;
    ssa.run(iterationTimes, attenuationRate, pchange, pmask);

    getchar();
    return 0;
}