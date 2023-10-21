#include "main_config.h"

#include "../utils.h"
#include "../multicore_experiments.h"
#include "../convex_hull.h"
#include <omp.h>

constexpr double detector_w = 1.;
constexpr int TRIALS_PER_CONFIG = 100;
constexpr int CONFIGS_PER_EXPOSURE_TIME_SAMPLE = 4;
constexpr int EXPOSURE_TIME_SAMPLES = 4;

// 5625 total configs took about 5 minutes on the shitbox computer @ 250 trials per config
constexpr int TOTAL_CONFIGS = CONFIGS_PER_EXPOSURE_TIME_SAMPLE * EXPOSURE_TIME_SAMPLES;
constexpr double PREDICTED_TIME_EB = 5. * (double)TOTAL_CONFIGS / 5625. * (double)TRIALS_PER_CONFIG / (double)250;
constexpr int TOTAL_SAMPLES = TRIALS_PER_CONFIG * TOTAL_CONFIGS;

constexpr double MIN_EXPOSURE_TIME = 10;
constexpr double MAX_EXPOSURE_TIME = 100000;

constexpr CHI2_METHOD chi_2_method = NORMALIZE;

constexpr double MAX_R = 0.5;

void mainNoiseResponse(void)
{
    printf("> %s\n", (chi_2_method == NORMALIZE ? "normalized chi2" : "non-normalized chi2"));

    ArrX3d core_locations;
    Eigen::VectorXi g2_capable_idx;
    int num_cores;

    g2_capable_idx = Eigen::VectorXi(3,1);

    g2_capable_idx(0) = 1;
    g2_capable_idx(1) = 3;
    g2_capable_idx(2) = 5;

    createConcentricCores(core_locations, 1, 1.);

    // createWorboyCores(core_locations, g2_capable_idx);
   
    savePoints("finite-time/core_locations.csv", core_locations);
    saveIndexes("finite-time/g2_capable_indexes.csv", g2_capable_idx);

    Eigen::Array<double,CONFIGS_PER_EXPOSURE_TIME_SAMPLE,EXPOSURE_TIME_SAMPLES> w_eff_bar;
    Eigen::Array<double,TOTAL_CONFIGS,5> emitter_parameter_log;

    printf("> Predicting a run time of %f minutes (EB)\n", PREDICTED_TIME_EB);
    
    auto start = high_resolution_clock::now();

    // omp_set_num_threads(1);
    // #pragma omp parallel
    // for (int linear_sample_idx = omp_get_thread_num(); linear_sample_idx < TOTAL_CONFIGS; linear_sample_idx += omp_get_num_threads())
    for (int linear_sample_idx = 0; linear_sample_idx < TOTAL_CONFIGS; linear_sample_idx++)
    {
        int noise_sample_idx = linear_sample_idx % EXPOSURE_TIME_SAMPLES;
        int config_sample_Idx = linear_sample_idx / EXPOSURE_TIME_SAMPLES;

        printf("> started %i,%i (%i/%i)\n", noise_sample_idx, config_sample_Idx, linear_sample_idx, TOTAL_CONFIGS);

        double exposure_time = MIN_EXPOSURE_TIME + (MAX_EXPOSURE_TIME - MIN_EXPOSURE_TIME) * (double)noise_sample_idx / double(EXPOSURE_TIME_SAMPLES - 1);


        Eigen::Array<double,2,3> emitter_xy = Eigen::Array<double,2,3>::Random();

        Eigen::Array<double,1,3> emitter_offset = Eigen::Array<double,1,3>::Random();

        emitter_xy.row(0) += emitter_offset;
        emitter_xy.row(1) += emitter_offset;

        emitter_xy.col(2) = 0;

        double r = (emitter_xy.row(0) - emitter_xy.row(1)).matrix().norm();

        if ( r > MAX_R )
        {
            Eigen::Array3d center = 0.5 * (emitter_xy.row(0) + emitter_xy.row(1));

            emitter_xy.row(0) -= center;
            emitter_xy.row(1) -= center;

            emitter_xy.row(0) *= MAX_R / r;
            emitter_xy.row(1) *= MAX_R / r;

            emitter_xy.row(0) += center;
            emitter_xy.row(1) += center;
        }

        emitter_parameter_log(linear_sample_idx,0) = emitter_xy(0,0);
        emitter_parameter_log(linear_sample_idx,1) = emitter_xy(0,1);
        emitter_parameter_log(linear_sample_idx,2) = emitter_xy(1,0);
        emitter_parameter_log(linear_sample_idx,3) = emitter_xy(1,1);

        Eigen::Array<double,2,1> emitter_brightness = Eigen::Array<double,2,1>::Random();

        emitter_brightness[0] = 1.;

        emitter_brightness[1] *= 0.5;
        emitter_brightness[1] += 0.5;

        exposure_time /= emitter_brightness[1];
        printf("> exposure time: %f\n", exposure_time);

        emitter_parameter_log(linear_sample_idx,4) = emitter_brightness[0] / emitter_brightness[1];

        ArrX2d x1s = ArrX2d(TRIALS_PER_CONFIG,2);
        ArrX2d x2s = ArrX2d(TRIALS_PER_CONFIG,2);

        Eigen::Array<double,Eigen::Dynamic,1> p02s = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);
        Eigen::Array<double,Eigen::Dynamic,1> chi2 = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);

        printf("here2\n");

        std::cout << emitter_xy << std::endl;

        #pragma omp parallel
        // for (int cts =0; cts < TRIALS_PER_CONFIG; cts++)
        for (int cts = omp_get_thread_num(); cts < TRIALS_PER_CONFIG; cts += omp_get_num_threads())
        {
            ArrX2d multicore_measure;

            bool valid = false;

            while (!valid)
            {
                ArrX2d multicore_measure = multicoreMeasureFiniteTime(
                    core_locations,
                    g2_capable_idx,
                    emitter_xy,
                    emitter_brightness,
                    exposure_time
                );

                // valid = true;

                valid = !multicore_measure.hasNaN();

                // for (int idx = 0; idx < multicore_measure.rows(); idx++)
                // {
                //     if multicore
                // }
            }

            // std::cout << multicore_measure << std::endl;

            Eigen::VectorXd xx = 4. * Eigen::Array<double,5,1>::Random(5,1);

            xx(4) = 0.5;

            MulticoreDataFiniteTime mc_data = {
                core_locations, multicore_measure, g2_capable_idx, 100000000, chi_2_method
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

        printf("here1\n");

        ArrX2d thresholded_x1s = thresholdGuesses(x1s, 1 - 1./sqrt(exp(1.)));
        ArrX2d x1s_convex_hull = convexHull(thresholded_x1s);
        double x1s_cvx_hull_area = polygonArea(x1s_convex_hull);
        double e1_weff = 2. * sqrt(x1s_cvx_hull_area / PI);

        ArrX2d thresholded_x2s = thresholdGuesses(x2s, 1 - 1./sqrt(exp(1.)));
        ArrX2d x2s_convex_hull = convexHull(thresholded_x2s);
        double x2s_cvx_hull_area = polygonArea(x2s_convex_hull);
        double e2_weff = 2. * sqrt(x2s_cvx_hull_area / PI);

        w_eff_bar(config_sample_Idx, noise_sample_idx) = 0.5 * (e1_weff + e2_weff);

        // std::cout << noise_sample_idx << ", " << config_sample_Idx << ", " << noise << std::endl;
        printf("> stopped %i,%i (%i/%i)\n", noise_sample_idx, config_sample_Idx, linear_sample_idx, TOTAL_CONFIGS);
    }

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(stop - start);

    std::cout << "> Completed in " << (double)duration.count() / 60. << " minutes." << std::endl;

    savePoints("finite-time/emitter_parameter_log.csv", emitter_parameter_log);
    saveHeatmap("finite-time/w_eff_bar.csv", w_eff_bar);
}