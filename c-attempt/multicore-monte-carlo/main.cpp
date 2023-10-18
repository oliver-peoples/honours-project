#include <Eigen/Dense>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string.h>
#include "../fast_gp.h"
#include "../utils.h"
#include <iostream>
#include <omp.h>

#define USE_OMP 0

#include <chrono>
using namespace std::chrono;

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>

constexpr double detector_w = 1.;

constexpr int TRIALS_PER_CONFIG = 300;

constexpr int X_SAMPLES = 40;
constexpr int Y_SAMPLES = 40;

constexpr int PER_PIXEL_CONFIGS = 10;

constexpr int total_samples = TRIALS_PER_CONFIG * PER_PIXEL_CONFIGS * X_SAMPLES * Y_SAMPLES;

constexpr double PM = 1;

int main()
{
    ArrX3d core_locations;
    int num_cores;

    createConcentricCores(core_locations, 1, 1.);
    
    savePoints("multicore-monte-carlo/core_locations.csv", core_locations);

    Eigen::VectorXi G2_CAPABLE_IDX = Eigen::VectorXi(3,1);

    G2_CAPABLE_IDX(0) = 1;
    G2_CAPABLE_IDX(1) = 3;
    G2_CAPABLE_IDX(2) = 5;

    saveIndexes("multicore-monte-carlo/g2_capable_indexes.csv", G2_CAPABLE_IDX);

    Eigen::Array<double,X_SAMPLES,Y_SAMPLES> e1_weffs;
    Eigen::Array<double,X_SAMPLES,Y_SAMPLES> e2_weffs;
    Eigen::Array<double,X_SAMPLES,Y_SAMPLES> avg_weffs;

    #if USE_OMP
    #pragma omp parallel
    for (int linear_pix_idx = omp_get_thread_num(); linear_pix_idx < X_SAMPLES * Y_SAMPLES; linear_pix_idx += omp_get_num_threads())
    #else
    for (int linear_pix_idx = 0; linear_pix_idx < X_SAMPLES * Y_SAMPLES; linear_pix_idx++)
    #endif
    {
        int x_idx = linear_pix_idx % X_SAMPLES;
        int y_idx = linear_pix_idx % Y_SAMPLES;

        #if !USE_OMP
        std::cout << x_idx << "," << y_idx << std::endl;
        #endif

        double x_pos = -PM + 2. * (double)x_idx / double(X_SAMPLES - 1);
        double y_pos = PM - 2. * (double)x_idx / double(Y_SAMPLES - 1);

        for (int config_idx = 0; config_idx < PER_PIXEL_CONFIGS; config_idx++)
        {
            Eigen::Array<double,2,3> emitter_xy = Eigen::Array<double,2,3>::Random();

            emitter_xy.col(0) *= 0.25;
            emitter_xy.col(1) *= 0.25;

            double mean_x = emitter_xy.col(0).mean();
            double mean_y = emitter_xy.col(1).mean();

            double mean_x_diff = x_pos - mean_x;
            double mean_y_diff = y_pos - mean_y;

            emitter_xy.col(0) += mean_x_diff;
            emitter_xy.col(1) += mean_y_diff;
            emitter_xy(0,2) = 1.;
            emitter_xy(1,2) *= 0.5;
            emitter_xy(1,2) += 0.5;

            Eigen::Array<double,2,1> emitter_brightness = emitter_xy.col(2);

            emitter_xy.col(2) *= 0;

            // savePoints("multicore-monte-carlo/emitter_xy.csv", emitter_xy);

            ArrX2d multicore_measure = multicoreMeasure(
                core_locations,
                G2_CAPABLE_IDX,
                emitter_xy,
                emitter_brightness
            );

            // saveMeasurements("multicore-monte-carlo/g1_g2_measurements.csv", multicore_measure);

            ArrX2d x1s = ArrX2d(TRIALS_PER_CONFIG,2);
            ArrX2d x2s = ArrX2d(TRIALS_PER_CONFIG,2);
            Eigen::Array<double,Eigen::Dynamic,1> p02s = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);
            Eigen::Array<double,Eigen::Dynamic,1> chi2 = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);

            double variab = 0.1;

            // auto start = high_resolution_clock::now();

            for (int cts = 0; cts < TRIALS_PER_CONFIG; cts++)
            {
                ArrX2d multicore_measure_noisy = multicore_measure;

                bool sane_results = 0;

                Eigen::VectorXd xx = Eigen::Array<double,5,1>::Random(5,1);

                int fail_counts = 0;

                while(!sane_results && fail_counts < 10)
                {

                    multicore_measure_noisy.col(0) *= (1 + variab * Eigen::Array<double,Eigen::Dynamic,1>::Random(multicore_measure.rows(),1));
                    multicore_measure_noisy(G2_CAPABLE_IDX,1) *= (1 + variab * Eigen::Array<double,Eigen::Dynamic,1>::Random(G2_CAPABLE_IDX.rows(),1));

                    xx(4) = 0.5;

                    MulticoreData mc_data = {
                        core_locations, multicore_measure_noisy, G2_CAPABLE_IDX
                    };

                    bool success = optim::nm(xx, multicoreChi2, (void*)&mc_data);

                    if (xx.array().abs().maxCoeff() < 10)
                    {
                        sane_results = 1;

                        // std::cout << xx.transpose() << std::endl;
                    }
                    else
                    {
                        fail_counts++;
                        std::cout << "> " << cts << " fuck" << std::endl;
                    }
                }

                if (fail_counts < 10)

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

            // auto stop = high_resolution_clock::now();

            // auto duration = duration_cast<milliseconds>(stop - start);

            // std::cout << duration.count() << std::endl;

            ArrX2d thresholded_x1s = thresholdGuesses(x1s, 1 - 1./sqrt(exp(1.)));
            ArrX2d x1s_convex_hull = convexHull(thresholded_x1s);
            double x1s_cvx_hull_area = polygonArea(x1s_convex_hull);
            double e1_weff = 2. * sqrt(x1s_cvx_hull_area / PI);

            std::cout << "here" << std::endl;
            std::cout << x2s << std::endl;
            ArrX2d thresholded_x2s = thresholdGuesses(x2s, 1 - 1./sqrt(exp(1.)));
            ArrX2d x2s_convex_hull = convexHull(thresholded_x2s);
            double x2s_cvx_hull_area = polygonArea(x2s_convex_hull);
            double e2_weff = 2. * sqrt(x2s_cvx_hull_area / PI);
            // double x2s_cvx_hull_area = polygonArea(x2s_convex_hull);
            // double e2_weff = 2. * sqrt(x2s_cvx_hull_area / PI);

            // std::cout << e2_weff << std::endl;

            // e1_weffs(y_idx,x_idx) = e1_weff;
            // e2_weffs(y_idx,x_idx) = e2_weff;
            // avg_weffs(y_idx,x_idx) = 0.5 * (e1_weff + e2_weff);
            
            // savePoints("multicore-monte-carlo/x1s_convex_hull.csv", x1s_convex_hull);
            // savePoints("multicore-monte-carlo/x2s_convex_hull.csv", x2s_convex_hull);

            // savePoints("multicore-monte-carlo/x1s.csv", x1s);
            // savePoints("multicore-monte-carlo/x2s.csv", x2s);

            // printf("> fit quality emitter 1: %f, %f\n", fitQuality(x1s), e1_weff);
            // printf("> fit quality emitter 2: %f, %f\n", fitQuality(x2s), e2_weff);
        }
    }

    saveHeatmap("multicore-monte-carlo/e1_weffs.csv", e1_weffs);
    saveHeatmap("multicore-monte-carlo/e2_weffs.csv", e2_weffs);
    saveHeatmap("multicore-monte-carlo/avg_weffs.csv", avg_weffs);

    return 0;
}