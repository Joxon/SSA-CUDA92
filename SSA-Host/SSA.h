#ifndef SSA_SSA_H
#define SSA_SSA_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <chrono>
#include <vector>

class Problem
{
public:
    //问题的维度
    unsigned int dimension;

    Problem(unsigned int dimension) : dimension(dimension) {}

    virtual double getFitness(const std::vector<double>& solution) = 0;
};

class Position
{
public:
    //位置的适应度
    double fitness;
    //蜘蛛的移动策略，是一个与待求解问题维度相同的向量
    std::vector<double> solution;

    Position() {};
    Position(const std::vector<double>& solution, double fitness) :
        solution(solution), fitness(fitness)
    {}

    friend bool operator==(const Position& p1, const Position& p2)
    {
        for(int i = 0; i < p1.solution.size(); ++i)
        {
            if(p1.solution[i] != p2.solution[i])
            {
                return false;
            }
        }
        return true;
    }

    friend double operator-(const Position& p1, const Position& p2)
    {
        double distance = 0.0;
        for(int i = 0; i < p1.solution.size(); ++i)
        {
            distance += fabs(p1.solution[i] - p2.solution[i]);
        }
        return distance;
    }

    //初始化位置
    static Position init_position(Problem* problem);
};

class Vibration
{
public:
    //蜘蛛产生的震动强度，大于等于0
    double intensity;
    //当前最佳位置
    Position position;
    //C为常数
    static double C;

    Vibration() {}
    Vibration(const Position& position);
    Vibration(double intensity, const Position& position);

    //震动强度衰减
    double intensity_attenuation(double attenuation_factor, double distance) const;
    //自身震动强度，fitness是目标源所处位置的适应度
    static double fitness_to_intensity(double fitness);
};

class Spider
{
public:
    //当前位置
    Position position;
    //上一轮迭代的目标震动强度
    Vibration target_vibr;
    //维度遮罩？
    std::vector<bool> dimension_mask;
    //上一次移动？
    std::vector<double> previous_move;
    //不活动程度？
    int inactive_deg;

    Spider(const Position& position);

    void choose_vibration(const std::vector<Vibration>& vibrations,
                          const std::vector<double>& distances, double attenuation_factor);
    void mask_changing(double p_change, double p_mask);
    void random_walk(const std::vector<Vibration>& vibrations);
};

//SSA算法本身
class SSA
{
public:
    //待求解问题
    Problem * problem;
    //问题维度
    unsigned int dimension;
    //蜘蛛种群
    std::vector<Spider> population;
    //所有震动
    std::vector<Vibration> vibrations;
    //二维距离列表？
    std::vector<std::vector<double>> distances;
    //全局最优位置
    Position global_best_position;

    SSA(Problem* problem, unsigned int pop_size);

    //适应度计算
    void fitness_calculation();
    //产生震动
    void vibration_generation(double attenuation_rate);
    //算法执行
    void run(int max_iteration, double attenuation_rate, double p_change, double p_mask);

private:
    //迭代次数
    int iteration;
    //种群最优适应度
    double population_best_fitness;
    //衰减基准
    double attenuation_base;
    //平均距离
    double mean_distance;
    //算法开始执行的时刻
    std::chrono::high_resolution_clock::time_point start_time;
    //结果打印
    void print_header();
    void print_content();
    void print_footer();
};

//求平均
inline double mean(std::vector<double> data)
{
    double sum = 0.0;
    for(int i = 0; i < data.size(); ++i)
    {
        sum += data[i];
    }
    return sum / data.size();
}

//求标准差，瓶颈
inline double std_dev(std::vector<double> data)
{
    double mean_val = mean(data);
    double sum = 0.0;
    for(int i = 0; i < data.size(); ++i)
    {
        sum += (mean_val - data[i]) * (mean_val - data[i]);
    }
    return sqrt(sum / data.size());
}

//求随机数
//rand(): 0 to RAND_MAX
//RAND_MAX = 32767
//RAND_MAX + 1.0 = 32768.0
//randu(): 0 to 1.0
inline double randu()
{
    return (double)rand() / (RAND_MAX + 1.0);
}

std::string get_time_string();
std::string get_time_string(std::chrono::milliseconds ms);

#endif