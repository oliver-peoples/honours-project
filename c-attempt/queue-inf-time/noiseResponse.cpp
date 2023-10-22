#include "main_config.h"
#include <cstdlib>

void noiseResponse(SimConfig sim_config)
{
    ArrX3d core_locations = sim_config.core_locations;
    Eigen::VectorXi g2_capable_idx = sim_config.g2_capable_idx();
    double detector_w = sim_config.detector_w;
    int TRIALS_PER_CONFIG = sim_config.TRIALS_PER_CONFIG;
    int CONFIGS_PER_NOISE_SAMPLE = sim_config.CONFIGS_PER_NOISE_SAMPLE;
    int NOISE_SAMPLES = sim_config.NOISE_SAMPLES;
    double MAX_R = sim_config.MAX_R;
    double MAX_NOISE_PCT = sim_config.MAX_NOISE_PCT;
    CHI2_METHOD chi_2_method = sim_config.chi_2_method;

    std::string save_path = sim_config.save_path;

    system(("mkdir queue-inf-time/" + save_path).c_str());

    int TOTAL_CONFIGS = sim_config.TOTAL_CONFIGS();
    int TOTAL_SAMPLES = sim_config.TOTAL_SAMPLES();

    printf("> %s\n", (chi_2_method == NORMALIZE ? "normalized chi2" : "non-normalized chi2"));
   
    savePoints("queue-inf-time/" + save_path + "/core_locations.csv", core_locations);
    saveIndexes("queue-inf-time/" + save_path + "/g2_capable_indexes.csv", g2_capable_idx);

    Eigen::ArrayXXd w_eff_bar = Eigen::ArrayXXd(CONFIGS_PER_NOISE_SAMPLE,NOISE_SAMPLES);
    Eigen::ArrayXXd emitter_parameter_log = Eigen::ArrayXXd(TOTAL_CONFIGS,5);

    // printf("> Predicting a run time of %f minutes (EB)\n", PREDICTED_TIME_EB);
    
    auto start = high_resolution_clock::now();

    #pragma omp parallel
    for (int linear_sample_idx = omp_get_thread_num(); linear_sample_idx < TOTAL_CONFIGS; linear_sample_idx += omp_get_num_threads())
    {
        int noise_sample_idx = linear_sample_idx % NOISE_SAMPLES;
        int config_sample_Idx = linear_sample_idx / NOISE_SAMPLES;

        if (config_sample_Idx == 0)
        {
            printf("> %i,%i (%i/%i)\n", noise_sample_idx, config_sample_Idx, linear_sample_idx, TOTAL_CONFIGS);
        }

        double noise = 0.01 * MAX_NOISE_PCT * (double)noise_sample_idx / double(NOISE_SAMPLES - 1);

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

        emitter_parameter_log(linear_sample_idx,4) = emitter_brightness[0] / emitter_brightness[1];

        ArrX2d multicore_measure = multicoreMeasureInfTime(
            core_locations,
            g2_capable_idx,
            emitter_xy,
            emitter_brightness
        );

        ArrX2d x1s = ArrX2d(TRIALS_PER_CONFIG,2);
        ArrX2d x2s = ArrX2d(TRIALS_PER_CONFIG,2);

        Eigen::Array<double,Eigen::Dynamic,1> p02s = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);
        Eigen::Array<double,Eigen::Dynamic,1> chi2 = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);

        for (int cts = 0; cts < TRIALS_PER_CONFIG; cts++)
        {
            ArrX2d multicore_measure_noisy = multicore_measure;

            multicore_measure_noisy.col(0) *= (1 + noise * Eigen::Array<double,Eigen::Dynamic,1>::Random(multicore_measure.rows(),1));
            multicore_measure_noisy(g2_capable_idx,1) *= (1 + noise * Eigen::Array<double,Eigen::Dynamic,1>::Random(g2_capable_idx.rows(),1));

            Eigen::VectorXd xx = Eigen::Array<double,5,1>::Random(5,1);

            xx(4) = 0.5;

            MulticoreDataInfTime mc_data = {
                core_locations, multicore_measure_noisy, g2_capable_idx, chi_2_method
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
    }

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(stop - start);

    std::cout << "> Completed " << sim_config.save_path << " in " << (double)duration.count() / 60. << " minutes." << std::endl;

    savePoints("queue-inf-time/" + save_path + "/emitter_parameter_log.csv", emitter_parameter_log);
    saveHeatmap("queue-inf-time/" + save_path + "/w_eff_bar.csv", w_eff_bar);

    system(("python queue-inf-time/plot_noise_response.py " + save_path).c_str());
}