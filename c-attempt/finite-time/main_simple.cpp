#include "main_config.h"

#include "../utils.h"
#include "../multicore_experiments.h"
#include "../convex_hull.h"
#include "misc/optim_structs.hpp"

constexpr double detector_w = 1.;
constexpr int TRIALS_PER_CONFIG = 500;

constexpr CHI2_METHOD chi_2_method = WORBOY;

// spectrum conjuction analysis tool (???)

void mainSimple(void)
{
    ArrX3d core_locations;
    Eigen::VectorXi g2_capable_idx;
    int num_cores;

    // g2_capable_idx = Eigen::VectorXi(6,1);

    // g2_capable_idx(0) = 1;
    // g2_capable_idx(1) = 2;
    // g2_capable_idx(2) = 3;
    // g2_capable_idx(3) = 4;
    // g2_capable_idx(4) = 5;
    // g2_capable_idx(5) = 6;

    g2_capable_idx = Eigen::VectorXi(4,1);

    g2_capable_idx(0) = 1;
    g2_capable_idx(1) = 3;
    g2_capable_idx(2) = 5;
    g2_capable_idx(3) = 0;

    // g2_capable_idx = Eigen::VectorXi(3,1);

    // g2_capable_idx(0) = 0;
    // g2_capable_idx(1) = 1;
    // g2_capable_idx(2) = 2;


    // g2_capable_idx = Eigen::VectorXi(6,1);

    // g2_capable_idx(0) = 1;
    // g2_capable_idx(1) = 3;
    // g2_capable_idx(2) = 5;
    // g2_capable_idx(3) = 7+2;
    // g2_capable_idx(4) = 11+2;
    // g2_capable_idx(5) = 15+2;

    createConcentricCores(core_locations, 1, 2.);

    // createWorboyCores(core_locations, g2_capable_idx);
   
    savePoints("finite-time/core_locations.csv", core_locations);
    saveIndexes("finite-time/g2_capable_indexes.csv", g2_capable_idx);

    // std::cout << g2_capable_idx << std::endl;

    Eigen::Array<double,2,3> emitter_xy {
        { -0.6300,-0.1276,0 },
        { 0.5146,-0.5573,0 }
    };

    // Eigen::Array<double,2,3> emitter_xy {
    //     { 1.55076,-1.05042,0 },
    //     { 1.05196,-1.01593,0 }
    // };

    // Eigen::Array<double,2,3> emitter_xy {
    //     { -0.1,-0.1,0 },
    //     { 0.1,0.1,0 }
    // };


    // emitter_xy *= 0.6;

    // emitter_xy += 0.25 * Eigen::Array<double,2,3>::Random();

    emitter_xy.col(2) = 0;

    savePoints("finite-time/emitter_xy.csv", emitter_xy);

    Eigen::Array<double,2,1> emitter_brightness {
        1.,
        0.3617
    };

    double t = 10000. / (emitter_brightness[1]);

    ArrX2d x1s = ArrX2d(TRIALS_PER_CONFIG,2);
    ArrX2d x2s = ArrX2d(TRIALS_PER_CONFIG,2);
    Eigen::Array<double,Eigen::Dynamic,1> p02s = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);
    Eigen::Array<double,Eigen::Dynamic,1> chi2 = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);

    auto start = high_resolution_clock::now();

    std::default_random_engine generator(std::random_device{}());

    #pragma omp parallel
    for (int cts = omp_get_thread_num(); cts < TRIALS_PER_CONFIG; cts += omp_get_num_threads())
    {
        // std::cout << cts << std::endl;

        // ArrX2d multicore_measure = multicoreMeasureFiniteTime(
        //     core_locations,
        //     g2_capable_idx,
        //     emitter_xy,
        //     emitter_brightness,
        //     t
        // );

        ArrX2d multicore_measure = multicoreMeasureFiniteTime(
            core_locations,
            g2_capable_idx,
            emitter_xy,
            emitter_brightness,
            t,
            &generator
        );

        //     // valid = true;

        //     valid = !multicore_measure.hasNaN();

        //     // for (int idx = 0; idx < multicore_measure.rows(); idx++)
        //     // {
        //     //     if multicore
        //     // }
        // }
        
        // std::cout << multicore_measure << std::endl;

        // std::cout << multicore_measure << std::endl;

        Eigen::VectorXd xx = Eigen::Array<double,5,1>::Random(5,1);

        xx(4) = 0.5;

        MulticoreDataFiniteTime mc_data = {
            core_locations, multicore_measure, g2_capable_idx, 100000000, chi_2_method, &generator
        };

        bool success = optim::nm(xx, multicoreFiniteTimeChi2, (void*)&mc_data);

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

    double mean_e1_x = thresholded_x1s.col(0).mean();
    double mean_e1_error_x = thresholded_x1s.col(0).mean() - emitter_xy(0,0);
    double mean_e1_y = thresholded_x1s.col(1).mean();
    double mean_e1_error_y = thresholded_x1s.col(1).mean() - emitter_xy(0,1);

    ArrX2d thresholded_x2s = thresholdGuesses(x2s, 1 - 1./sqrt(exp(1.)));
    ArrX2d x2s_convex_hull = convexHull(thresholded_x2s);
    double x2s_cvx_hull_area = polygonArea(x2s_convex_hull);
    double e2_weff = 2. * sqrt(x2s_cvx_hull_area / PI);

    double mean_e2_x = thresholded_x2s.col(0).mean();
    double mean_e2_error_x = thresholded_x2s.col(0).mean() - emitter_xy(1,0);
    double mean_e2_y = thresholded_x2s.col(1).mean();
    double mean_e2_error_y = thresholded_x2s.col(1).mean() - emitter_xy(1,1);

    x1s_convex_hull = loopify(x1s_convex_hull);
    x2s_convex_hull = loopify(x2s_convex_hull);

    ArrX3d emitter_guesses = emitter_xy * 0;

    emitter_guesses(0,0) = mean_e1_x;
    emitter_guesses(0,1) = mean_e1_y;
    emitter_guesses(1,0) = mean_e2_x;
    emitter_guesses(1,1) = mean_e2_y;

    savePoints("finite-time/emitter_guesses.csv", emitter_guesses);
   
    savePoints("finite-time/x1s_convex_hull.csv", x1s_convex_hull);
    savePoints("finite-time/x2s_convex_hull.csv", x2s_convex_hull);

    savePoints("finite-time/x1s.csv", x1s);
    savePoints("finite-time/x2s.csv", x2s);

    printf("> %s\n", (chi_2_method == NORMALIZE ? "normalized chi2" : "non-normalized chi2"));
    printf("> fit quality emitter 1: %0.4f, %0.4f, %0.4f\n", fitQuality(x1s), e1_weff, sqrt(mean_e1_error_x * mean_e1_error_x + mean_e1_error_y * mean_e1_error_y));
    printf("> fit quality emitter 2: %0.4f, %0.4f, %0.4f\n", fitQuality(x2s), e2_weff, sqrt(mean_e2_error_x * mean_e2_error_x + mean_e2_error_y * mean_e2_error_y));
    printf("> fit quality overall: %0.4f, %0.4f, SF: %f\n", 0.5 * (fitQuality(x1s) + fitQuality(x2s)), 0.5 * (e1_weff + e2_weff), 1./ (0.5 * (e1_weff + e2_weff)));
    printf(
        "$%0.4f$ & $%0.4f$ & $%0.4f$ & $%0.4f$ & $%0.4f$ & $%0.4f$\n",
        fitQuality(x1s), e1_weff, sqrt(mean_e1_error_x * mean_e1_error_x + mean_e1_error_y * mean_e1_error_y),
        fitQuality(x2s), e2_weff, sqrt(mean_e2_error_x * mean_e2_error_x + mean_e2_error_y * mean_e2_error_y)
    );
}