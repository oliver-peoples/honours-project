#ifndef __MAIN_CONFIG_H__
#define __MAIN_CONFIG_H__

#include <Eigen/Dense>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <iostream>
#include <omp.h>
#include <vector>

#include "../utils.h"
#include "../multicore_experiments.h"
#include "../convex_hull.h"
#include <omp.h>

#include <chrono>
using namespace std::chrono;

#define OPTIM_USE_OPENMP
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>

struct SimConfig
{
    ArrX3d core_locations;
    std::vector<int> g2_capable_idx_vec;
    std::string save_path;
    double detector_w = 1.;
    int TRIALS_PER_CONFIG = 200;
    int CONFIGS_PER_NOISE_SAMPLE = 400;
    int NOISE_SAMPLES = 400;
    double MAX_R = 0.5;
    double MAX_NOISE_PCT = 20;
    CHI2_METHOD chi_2_method = NORMALIZE;

    int TOTAL_CONFIGS() { return this->CONFIGS_PER_NOISE_SAMPLE * this->NOISE_SAMPLES; }
    int TOTAL_SAMPLES() { return this->TRIALS_PER_CONFIG * this->TOTAL_CONFIGS(); }

    Eigen::VectorXi g2_capable_idx()
    {
        Eigen::VectorXi vec = Eigen::VectorXi(this->g2_capable_idx_vec.size());

        for (int idx = 0; idx < this->g2_capable_idx_vec.size(); idx++)
        {
            vec[idx] = this->g2_capable_idx_vec[idx];
        }

        return vec;
    }
};

#endif /* __ADIRS_MAIN_CONFIG_H__ */
