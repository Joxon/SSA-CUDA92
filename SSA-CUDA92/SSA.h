#pragma once

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <chrono>
#include <vector>

#include "SSA-cuda.h"

curandState * dStates;

class Problem
{
public:
    unsigned int dimension;

    Problem(unsigned int dimension) : dimension(dimension) {}

#if defined GET_FITNESS_THRUST
    virtual float getFitnessThrust(const std::vector<float>& solution) = 0;

#elif defined GET_FITNESS_CUDA
    virtual float getFitnessCuda(const std::vector<float>& solution) = 0;

#elif defined GET_FITNESS_HOST
    virtual float getFitnessHost(const std::vector<float>& solution) = 0;

#endif // GET_FITNESS_THRUST
};

class Position
{
public:
    float fitness;
    std::vector<float> solution; // solution.size() == problemDimension

    Position() {};
    Position(const std::vector<float>& solution, float fitness) :
        solution(solution), fitness(fitness)
    {}

    friend bool operator==(const Position& p1, const Position& p2)
    {
        for (int i = 0; i < p1.solution.size(); ++i)
        {
            if (p1.solution[i] != p2.solution[i])
            {
                return false;
            }
        }
        return true;
    }

    friend float operator-(const Position& p1, const Position& p2)
    {
        float distance = 0.0;
        for (int i = 0; i < p1.solution.size(); ++i)
        {
            distance += fabs(p1.solution[i] - p2.solution[i]);
        }
        return distance;
    }

    static Position initPosition(Problem* problem);
};

class Vibration
{
public:
    float intensity;
    Position position;
    static float C;

    Vibration() {}
    Vibration(const Position& position);
    Vibration(float intensity, const Position& position);

    float intensityAttenuation(float attenuation_factor, float distance) const;
    static float fitnessToIntensity(float fitness);
};

class Spider
{
public:
    Position position;
    Vibration targetVibr;
    std::vector<bool> dimensionMask;
    std::vector<float> previousMove;
    int inactiveDegree;

    Spider(const Position& position);

    void chooseVibrationHost(const std::vector<Vibration>& vibrations,
                             const std::vector<float>& distances, float attenuation_factor);
    void maskChanging(float p_change, float p_mask);
    void randomWalk(const std::vector<Vibration>& vibrations);
};

class SSA
{
public:
    Problem * problem;
    unsigned int dimension;
    std::vector<Spider> population; // size == populationSize
    std::vector<Vibration> vibrations; // size == populationSize
    std::vector<std::vector<float>> distances; // size == populationSize * populationSize

    Position globalBestPosition;

    SSA(Problem* problem, unsigned int popSize);

    void run(int maxIteration, float attenuationRate, float pChange, float pMask);
    void fitnessCalculation();
    void vibrationGeneration(float attenuationRate);

private:
    int iteration;
    float populationBestFitness;
    float attenuationBase;
    float meanDistance;

    std::chrono::high_resolution_clock::time_point startTime;
    void printHeader();
    void printContent();
    void printFooter();
};

inline float getMean(std::vector<float> data)
{
    float sum = 0.0;
    for (int i = 0; i < data.size(); ++i)
    {
        sum += data[i];
    }
    return sum / data.size();
}

inline float getStdDev(std::vector<float> data)
{
    float mean = getMean(data);
    float sum = 0.0;
    for (int i = 0; i < data.size(); ++i)
    {
        sum += (mean - data[i]) * (mean - data[i]);
    }
    return sqrt(sum / data.size());
}

//rand(): 0 - RAND_MAX
//generating on the interval[0, 1)
inline float randu()
{
    return (float)rand() / (RAND_MAX + 1.0f);
}

std::string getTimeStdString();
std::string getTimeStdString(std::chrono::milliseconds ms);

//typedef struct
//{
//    float fitness;
//    unsigned index;
//} Fitness;
