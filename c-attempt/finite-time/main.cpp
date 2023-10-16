#include <Eigen/Dense>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include "../fast_gp.h"
#include "../utils.h"
#include <iostream>
#include <omp.h>

#include <chrono>
using namespace std::chrono;

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>

constexpr double detector_w = 1.;
constexpr int TRIALS = 10000;

int main()
{
    ArrX3d core_locations;
    int num_cores;

    createConcentricCores(core_locations, 1, 1.);
    
    savePoints("finite-time/core_locations.csv", core_locations);

    Eigen::VectorXi G2_CAPABLE_IDX = Eigen::VectorXi(3,1);

    G2_CAPABLE_IDX(0) = 1;
    G2_CAPABLE_IDX(1) = 3;
    G2_CAPABLE_IDX(2) = 5;

    saveIndexes("finite-time/g2_capable_indexes.csv", G2_CAPABLE_IDX);

    // std::cout << G2_CAPABLE_IDX << std::endl;

    Eigen::Array<double,2,3> emitter_xy {
        { -0.6300,-0.1276,0 },
        { 0.5146,-0.5573,0 }
    };

    emitter_xy *= 0.25;

    savePoints("finite-time/emitter_xy.csv", emitter_xy);

    Eigen::Array<double,2,1> emitter_brightness {
        1.,
        0.3167
    };

    ArrX2d multicore_measure = multicoreMeasure(
        core_locations,
        G2_CAPABLE_IDX,
        emitter_xy,
        emitter_brightness
    );

    saveMeasurements("finite-time/g1_g2_measurements.csv", multicore_measure);

    ArrX2d x1s = ArrX2d(TRIALS,2);
    ArrX2d x2s = ArrX2d(TRIALS,2);
    Eigen::Array<double,Eigen::Dynamic,1> p02s = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS,1);
    Eigen::Array<double,Eigen::Dynamic,1> chi2 = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS,1);

    double variab = 0.1;

    auto start = high_resolution_clock::now();

    #pragma omp parallel
    for (int cts = omp_get_thread_num(); cts < TRIALS; cts += omp_get_num_threads())
    {
        ArrX2d multicore_measure_noisy = multicore_measure;

        multicore_measure_noisy.col(0) *= (1 + variab * Eigen::Array<double,Eigen::Dynamic,1>::Random(multicore_measure.rows(),1));
        multicore_measure_noisy(G2_CAPABLE_IDX,1) *= (1 + variab * Eigen::Array<double,Eigen::Dynamic,1>::Random(G2_CAPABLE_IDX.rows(),1));

        Eigen::VectorXd xx = Eigen::Array<double,5,1>::Random(5,1);

        xx(4) = 0.5;

        MulticoreData mc_data = {
            core_locations, multicore_measure_noisy, G2_CAPABLE_IDX
        };

        bool success = optim::nm(xx, multicoreChi2, (void*)&mc_data);

        if ( xx(4) < 1 )
        {
            x1s(cts,1 - 1) = xx(1 - 1);
            x1s(cts,2 - 1) = xx(2 - 1);
            x2s(cts,1 - 1) = xx(3 - 1);
            x2s(cts,2 - 1) = xx(4 - 1);
            p02s(cts) = xx(5 - 1);
        }
        else
        {
            x1s(cts,1 - 1) = xx(3 - 1);
            x1s(cts,2 - 1) = xx(4 - 1);
            x2s(cts,1 - 1) = xx(1 - 1);
            x2s(cts,2 - 1) = xx(2 - 1);
            p02s(cts) = 1/xx(5 - 1);
        }

        // ArrX2d multicore_measure_noisy = multicore_measure * (1 + ArrX2d::Random())
    }

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << duration.count() << std::endl;

    ArrX2d x1s_convex_hull = convexHull(x1s);

    std::cout << x1s_convex_hull << std::endl;

    savePoints("finite-time/x1s_convex_hull.csv", x1s_convex_hull);

    savePoints("finite-time/x1s.csv", x1s);
    savePoints("finite-time/x2s.csv", x2s);

    printf("> fit quality emitter 1: %f\n", fitQuality(x1s));
    printf("> fit quality emitter 2: %f\n", fitQuality(x2s));

    return 0;
}