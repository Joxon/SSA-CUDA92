#include "SSA.h"

Position Position::init_position(Problem* problem)
{
    //初始化solution的大小等于问题的维度
    std::vector<double> solution(problem->dimension);
    //随机化solution中的值
    for (unsigned i = 0; i < problem->dimension; ++i)
    {
        solution[i] = randu() * 200.0f - 100.0f;
    }
    return Position(solution, 1E100);
}

Vibration::Vibration(const Position& position) :
    intensity(fitness_to_intensity(position.fitness)), position(position)
{}

Vibration::Vibration(double intensity, const Position& position) :
    intensity(intensity), position(position)
{}

Spider::Spider(const Position& position) :
    position(position), inactive_deg(0)
{
    target_vibr = Vibration(0, position);
    dimension_mask.resize(position.solution.size());
    previous_move.resize(position.solution.size());
}

SSA::SSA(Problem* problem, unsigned int pop_size) :
    problem(problem), dimension(problem->dimension)
{
    srand((unsigned int)(time(NULL) + (int)this));

    //reserve() does not create objects
    population.reserve(pop_size);
    vibrations.reserve(pop_size);

    //resize() does create objects
    distances.resize(pop_size);
    for (unsigned i = 0; i < pop_size; ++i)
    {
        Position position = Position::init_position(problem);
        population.push_back(Spider(position));
        distances[i].resize(pop_size);
    }

    global_best_position = population[0].position;
}

double Vibration::intensity_attenuation(double attenuation_factor, double distance) const
{
    return intensity * exp(-distance / attenuation_factor);
}

double Vibration::fitness_to_intensity(double fitness)
{
    return  log(1.0 / (fitness - C) + 1.0);
}

double Vibration::C = -1E-100;

void SSA::print_header()
{
    std::cout << "               SSA starts at "
        << get_time_string() << std::endl
        << "==============================================================" << std::endl
        << " iter    optimum    pop_min  base_dist  mean_dist time_elapsed" << std::endl
        << "==============================================================" << std::endl;
}

void SSA::print_content()
{
    auto current_time = std::chrono::high_resolution_clock::now();
    printf("%5d %.3e %.3e %.3e %.3e ", iteration, global_best_position.fitness, population_best_fitness,
           attenuation_base, mean_distance);
    std::cout << get_time_string(std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time)) << std::endl;
}

void SSA::print_footer()
{
    std::cout << "==============================================================" << std::endl;
    print_content();
    std::cout << "==============================================================" << std::endl;
}

std::string get_time_string()
{
    using namespace std::chrono;

    auto time_t = system_clock::to_time_t(system_clock::now());
    auto ttm = localtime(&time_t);

    char time_str[80];
    strftime(time_str, 80, "%Y-%m-%d %H:%M:%S", ttm);
    return std::string(time_str);
}

std::string get_time_string(std::chrono::milliseconds ms)
{
    long long int mss = ms.count();
    char time_str[80];
    sprintf(time_str, "%02d:%02d:%02d.%03d", int(mss / 3600000), int(mss / 60000 % 60),
            int(mss / 1000 % 60), int(mss % 1000));
    return std::string(time_str);
}

void SSA::run(int max_iteration,
              double attenuation_rate,
              double p_change,
              double p_mask)
{
    //打印表头
    print_header();
    //记录开始时刻
    start_time = std::chrono::high_resolution_clock::now();
    //开始迭代
    for (iteration = 1; iteration <= max_iteration; ++iteration)
    {
        //适应度计算，瓶颈！
        fitness_calculation();

        //产生震动，瓶颈！
        vibration_generation(attenuation_rate);

        //遍历种群中的每只蜘蛛
        for (int i = 0; i < population.size(); ++i)
        {
            //概率改变
            population[i].mask_changing(p_change, p_mask);

            //随机游走
            population[i].random_walk(vibrations);
        }

        //输出快照
        if ((iteration == 1) || (iteration == 10)
            || (iteration < 1001 && iteration % 100 == 0)
            || (iteration < 10001 && iteration % 1000 == 0)
            || (iteration < 100001 && iteration % 10000 == 0))
        {
            print_content();
        }
    }
    //打印表尾
    iteration--;
    print_footer();
}

//瓶颈
void SSA::fitness_calculation()
{
    population_best_fitness = 1E100;
    for (int i = 0; i < population.size(); ++i)
    {
        //计算当前的适应度
        //瓶颈，一个规约过程（求solution的平方和）
        double fitness = problem->getFitness(population[i].position.solution);

        //更新当前位置的适应度
        population[i].position.fitness = fitness;

        //适应度越小越好
        if (fitness < global_best_position.fitness)
        {
            global_best_position = population[i].position;
        }
        if (fitness < population_best_fitness)
        {
            population_best_fitness = fitness;
        }
    }
    mean_distance = 0;
    // Host gets all distances and their sum
    for (int i = 0; i < population.size(); ++i)
    {
        for (int j = i + 1; j < population.size(); ++j)
        {
            //friend float operator-(const Position& p1, const Position& p2)
            //{
            //    float distance = 0.0;
            //    for(int j = 0; j < p1.solution.size(); ++j)
            //    {
            //        distance += fabs(p1.solution[j] - p2.solution[j]);
            //    }
            //    return distance;
            //}

            //float distance = 0.0;
            //for(int k = 0; k < probDim; ++k)
            //{
            //    distance += fabs(population[j].position.solution[k] - population[j].position.solution[k]);
            //}
            distances[i][j] = population[i].position - population[j].position;//瓶颈
            distances[j][i] = distances[i][j];
            mean_distance += distances[i][j];
        }
    }
    mean_distance /= (population.size() * (population.size() - 1) / 2);
}

//瓶颈
void SSA::vibration_generation(double attenuation_rate)
{
    vibrations.clear();
    for (int i = 0; i < population.size(); ++i)
    {
        vibrations.push_back(std::move(Vibration(population[i].position)));
    }

    double sum = 0.0;
    std::vector<double> data;
    data.resize(population.size());
    for (unsigned i = 0; i < dimension; ++i)
    {
        //The standard deviation of all spiders' solutions
        for (int j = 0; j < population.size(); ++j)
        {
            data[j] = population[j].position.solution[i];
        }
        sum += std_dev(data);//瓶颈
    }

    attenuation_base = sum / dimension;
    for (int i = 0; i < population.size(); ++i)
    {
        population[i].choose_vibration(vibrations, distances[i], attenuation_base * attenuation_rate);//瓶颈
    }
    //bottleneck!!
    //population[j].choose_vibration(vibrations, distances[j], attenuation_factor);

    //Spider &spider = population[j];

    // Choose the max vibration
    //int max_index = -1;
    //float max_intensity = spider.target_vibr.intensity;

    // Host loop
    //for(int j = 0; j < vibrations.size(); ++j)
    //{
    //if(vibrations[j].position == spider.target_vibr.position)
    //{
    //    continue;
    //}
    //    bool isTargetVibr = true;
    //    for(int k = 0; k < vibrations[j].position.solution.size(); ++k)
    //    {
    //        if(vibrations[j].position.solution[k] !=
    //           spider.target_vibr.position.solution[k])
    //        {
    //            isTargetVibr = false;
    //            break;
    //        }
    //    }
    //    if(isTargetVibr)
    //    {
    //        continue;
    //    }

    //    //float intensity = vibrations[j].intensity_attenuation(attenuation_factor, distances[j][j]);
    //    float intensity = vibrations[j].intensity * exp((-distances[j][j]) / attenuation_factor);
    //    if(intensity > max_intensity)
    //    {
    //        max_index = j;
    //        max_intensity = intensity;
    //    }
    //}
}

//瓶颈
void Spider::choose_vibration(const std::vector<Vibration>& vibrations,
                              const std::vector<double>& distances,
                              double attenuation_factor)
{
    int max_index = -1;
    double max_intensity = target_vibr.intensity;
    for (int j = 0; j < vibrations.size(); ++j)
    {
        if (vibrations[j].position == target_vibr.position)
        {
            continue;
        }
        double intensity = vibrations[j].intensity_attenuation(attenuation_factor, distances[j]);
        if (intensity > max_intensity)
        {
            max_index = j;
            max_intensity = intensity;
        }
    }
    if (max_index != -1)
    {
        target_vibr = Vibration(max_intensity, vibrations[max_index].position);
        inactive_deg = 0;
    }
    else
    {
        ++inactive_deg;
    }
}

void Spider::mask_changing(double p_change, double p_mask)
{
    if (randu() > pow(p_change, inactive_deg))
    {
        inactive_deg = 0;
        p_mask *= randu();
        for (int i = 0; i < dimension_mask.size(); ++i)
        {
            dimension_mask[i] = (randu()) < p_mask;
        }
    }
}

void Spider::random_walk(const std::vector<Vibration>& vibrations)
{
    for (int i = 0; i < position.solution.size(); ++i)
    {
        previous_move[i] *= randu();

        double target_position = dimension_mask[i] ?
            vibrations[rand() % vibrations.size()].position.solution[i] :
            target_vibr.position.solution[i];

        previous_move[i] += randu() * (target_position - position.solution[i]);

        position.solution[i] += previous_move[i];
    }
}