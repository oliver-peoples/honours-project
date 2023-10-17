#include <eigen3/Eigen/Dense>

#define OPTIM_ENABLE_EIGEN_WRAPPERS

// #include <optim.hpp>

#include "fast_gp.h"

int main()
{  
    double (*tems[])(Eigen::Vector2d xy, double w, double I_0, Eigen::Vector2d tx, double r) = {
        &fastTEM_GH<0,0>,
        &fastTEM_GH<0,1>,
        &fastTEM_GH<1,0>,
        &fastTEM_GH<1,1>
    };

    Eigen::Vector2d illumination_centers[] = {
        { 0,0 },
        { 0,0 },
        { 0,0 },
        { 0,0 }
    };

    Eigen::Vector2d illumination_rotations[] = {
        { 0,0 },
        { 0,0 },
        { 0,0 },
        { 0,0 }
    };

    return 0;
}

