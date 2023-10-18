#include <Eigen/Dense>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include "../utils.h"
#include "../multicore_experiments.h"
#include "../convex_hull.h"
#include <iostream>
#include <omp.h>

#include <chrono>
using namespace std::chrono;

#define OPTIM_USE_OPENMP
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>

constexpr double detector_w = 1.;
constexpr int TRIALS_PER_CONFIG = 10000;
CHI2_METHOD chi_2_method = NORMALIZE;

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

    emitter_xy += 0.25 * Eigen::Array<double,2,3>::Random();

    emitter_xy.col(2) = 0;

    savePoints("finite-time/emitter_xy.csv", emitter_xy);

    Eigen::Array<double,2,1> emitter_brightness {
        1.,
        0.3167
    };

    ArrX2d multicore_measure = multicoreMeasureInfTime(
        core_locations,
        G2_CAPABLE_IDX,
        emitter_xy,
        emitter_brightness
    );

    saveMeasurements("finite-time/g1_g2_measurements.csv", multicore_measure);

    ArrX2d x1s = ArrX2d(TRIALS_PER_CONFIG,2);
    ArrX2d x2s = ArrX2d(TRIALS_PER_CONFIG,2);
    Eigen::Array<double,Eigen::Dynamic,1> p02s = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);
    Eigen::Array<double,Eigen::Dynamic,1> chi2 = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);

    double variab = 0.1;

    auto start = high_resolution_clock::now();

    #pragma omp parallel
    for (int cts = omp_get_thread_num(); cts < TRIALS_PER_CONFIG; cts += omp_get_num_threads())
    {
        ArrX2d multicore_measure_noisy = multicore_measure;

        multicore_measure_noisy.col(0) *= (1 + variab * Eigen::Array<double,Eigen::Dynamic,1>::Random(multicore_measure.rows(),1));
        multicore_measure_noisy(G2_CAPABLE_IDX,1) *= (1 + variab * Eigen::Array<double,Eigen::Dynamic,1>::Random(G2_CAPABLE_IDX.rows(),1));

        Eigen::VectorXd xx = Eigen::Array<double,5,1>::Random(5,1);

        xx(4) = 0.5;

        MulticoreDataInfTime mc_data = {
            core_locations, multicore_measure_noisy, G2_CAPABLE_IDX, chi_2_method
        };

        bool success = optim::nm(xx, multicoreInfTimeChi2, (void*)&mc_data);

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

    std::cout << duration.count() << ": " << (double)duration.count() / (double)TRIALS_PER_CONFIG << std::endl;

    ArrX2d thresholded_x1s = thresholdGuesses(x1s, 1 - 1./sqrt(exp(1.)));
    ArrX2d x1s_convex_hull = convexHull(thresholded_x1s);
    double x1s_cvx_hull_area = polygonArea(x1s_convex_hull);
    double e1_weff = 2. * sqrt(x1s_cvx_hull_area / PI);

    ArrX2d thresholded_x2s = thresholdGuesses(x2s, 1 - 1./sqrt(exp(1.)));
    ArrX2d x2s_convex_hull = convexHull(thresholded_x2s);
    double x2s_cvx_hull_area = polygonArea(x2s_convex_hull);
    double e2_weff = 2. * sqrt(x2s_cvx_hull_area / PI);

    x1s_convex_hull = loopify(x1s_convex_hull);
    x2s_convex_hull = loopify(x2s_convex_hull);
   
    savePoints("finite-time/x1s_convex_hull.csv", x1s_convex_hull);
    savePoints("finite-time/x2s_convex_hull.csv", x2s_convex_hull);

    savePoints("finite-time/x1s.csv", x1s);
    savePoints("finite-time/x2s.csv", x2s);

    printf("> fit quality emitter 1: %f, %f\n", fitQuality(x1s), e1_weff);
    printf("> fit quality emitter 2: %f, %f\n", fitQuality(x2s), e2_weff);
    printf("> fit quality overall: %f, %f\n", 0.5 * (fitQuality(x1s) + fitQuality(x2s)), 0.5 * (e1_weff + e2_weff));

    return 0;
}