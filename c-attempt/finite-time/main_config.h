#ifndef __MAIN_CONFIG_H__
#define __MAIN_CONFIG_H__

#include <Eigen/Dense>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <iostream>
#include <omp.h>

#include <chrono>
using namespace std::chrono;

#define OPTIM_USE_OPENMP
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>

#endif /* __MAIN_CONFIG_H__ */
