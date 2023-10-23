#include <Eigen/Dense>
#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
using namespace std::chrono;

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>

// #include <optim.hpp>

#include "../fast_gp.h"
#include "../utils.h"
#include "../tem_experiments.h"
#include "../convex_hull.h"

constexpr double wl = 633e-9;
constexpr double w0 = 633e-9;
constexpr double wn = 2 * PI / wl;

constexpr double detector_w = 1.;
constexpr int TRIALS_PER_CONFIG = 500;

constexpr CHI2_METHOD chi_2_method = NORMALIZE;

std::vector<Beam> beams = {
    // {
    //     fastTEM_GH<0,0>,
    //     { 0.0 * w0,0.0 * w0 },
    //     0,
    //     0,
    //     0
    // },
    // {
    //     fastTEM_GH<0,0>,
    //     1000000,
    //     { 0.0 * w0,1.0 * w0 },
    //     0,
    //     0,
    //     0,
    // },
    {
        fastTEM_GL<0,2>,
        2.*w0,
        { 0.0 * w0,1.0 * w0 },
        0,
        0,
        0
    },
    {
        fastTEM_GL<0,2>,
        2.*w0,
        { cos(PI * -30/180) * w0,sin(PI * -30/180) * w0 },
        0,
        0,
        0
    },
        {
        fastTEM_GL<0,2>,
        2.*w0,
        { -cos(PI * -30/180) * w0,sin(PI * -30/180) * w0 },
        0,
        -0,
        0
    },
    // {
    //     fastTEM_GH<1,0>,
    //     { 0.0 * w0,0.0 * w0 },
    //     0,
    //     0,
    //     0
    // },
    // {
    //     fastTEM_GH<1,1>,
    //     { 0.0 * w0,0.0 * w0 },
    //     0,
    //     0,
    //     0
    // },
    // {
    //     fastTEM_GH<1,2>,
    //     { 0.0 * w0,0.0 * w0 },
    //     0,
    //     0,
    //     0
    // },
    // {
    //     fastTEM_GH<2,1>,
    //     { 0.0 * w0,0.0 * w0 },
    //     0,
    //     0,
    //     0
    // }
    // {
    //     fastTEM_GH<2,2>,
    //     { 0.0 * w0,0.0 * w0 },
    //     0,
    //     0,
    //     0
    // }
    // {
    //     fastTEM_GL<1+0,2+1>,
    //     { 0.0 * w0,0.0 * w0 },
    //     0,
    //     0,
    //     0
    // },
    // {
    //     fastTEM_GL<1+1,2+0>,
    //     { 0.0 * w0,0.0 * w0 },
    //     0,
    //     0,
    //     0
    // },
    // {
    //     fastTEM_GL<1+1,2+1>,
    //     { 0.0 * w0,0.0 * w0 },
    //     0,
    //     0,
    //     0
    // },
};

int main()
{
    ArrX3d beam_centers = ArrX3d(beams.size(),3);

    for (int beam_idx = 0; beam_idx < beams.size(); beam_idx++)
    {
        beam_centers.row(beam_idx) = (1. / w0) * Eigen::Vector3d{ beams[beam_idx].center.x(),beams[beam_idx].center.y(),0. };
    }

    savePoints("tem-localization/beam_centers.csv", beam_centers);
    // position is in terms of w0

    Eigen::Array<double,2,3> emitter_xy {
        { -0.6300,-0.1276,0 },
        { 0.5146,-0.5573,0 }
    };

    // emitter_xy *= 0.25;
    
    Eigen::Array<double,2,1> emitter_brightness {
        1.,
        0.3617
    };

    emitter_xy.col(2) = 0;

    savePoints("tem-localization/emitter_xy.csv", emitter_xy);

    ArrX2d x1s = ArrX2d(TRIALS_PER_CONFIG,2);
    ArrX2d x2s = ArrX2d(TRIALS_PER_CONFIG,2);
    Eigen::Array<double,Eigen::Dynamic,1> p02s = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);
    Eigen::Array<double,Eigen::Dynamic,1> chi2 = Eigen::Array<double,Eigen::Dynamic,1>(TRIALS_PER_CONFIG,1);

    ArrX2d tem_measure = temMeasureInfTime(
        beams,
        emitter_xy,
        emitter_brightness,
        wl,
        w0,
        wn,
        detector_w
    );

    // std::cout << tem_measure << std::endl;

    saveMeasurements("tem-localization/g1_g2_measurements.csv", tem_measure);

    double variab = 0.1;
    double variab_2 = 1.;

    auto start = high_resolution_clock::now();

    #pragma omp parallel
    for (int cts = omp_get_thread_num(); cts < TRIALS_PER_CONFIG; cts += omp_get_num_threads())
    {
        ArrX2d tem_measure_noisy = tem_measure;

        tem_measure_noisy *= (1 + variab * Eigen::Array<double,Eigen::Dynamic,2>::Random(tem_measure.rows(),2));

        Eigen::VectorXd xx = Eigen::Array<double,5,1>::Random(5,1);
        // Eigen::VectorXd xx = Eigen::Array<double,5,1>{ emitter_xy(0,0),emitter_xy(0,1),emitter_xy(1,0),emitter_xy(1,1),emitter_brightness[1] };
        // xx.array() *= (1 + variab_2 * Eigen::Array<double,5,1>::Random());

        xx(4) = 0.5;

        TemDataInfTime tem_data = {
            beams, tem_measure_noisy, wl, w0, wn, detector_w, NORMALIZE
        };

        bool success = optim::nm(xx, temInfTimeChi2, (void*)&tem_data);

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

        // ArrX2d tem_measure_noisy = tem_measure * (1 + ArrX2d::Random())
    }

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << duration.count() << ": " << (double)duration.count() / (double)TRIALS_PER_CONFIG << std::endl;

    ArrX2d thresholded_x1s = thresholdGuesses(x1s, 1 - 1./sqrt(exp(1.)));
    ArrX2d x1s_convex_hull = convexHull(thresholded_x1s);
    double x1s_cvx_hull_area = polygonArea(x1s_convex_hull);
    double e1_weff = 2. * sqrt(x1s_cvx_hull_area / PI);

    double mean_e1_error_x = thresholded_x1s.col(0).mean() - emitter_xy(0,0);
    double mean_e1_error_y = thresholded_x1s.col(1).mean() - emitter_xy(0,1);

    ArrX2d thresholded_x2s = thresholdGuesses(x2s, 1 - 1./sqrt(exp(1.)));
    ArrX2d x2s_convex_hull = convexHull(thresholded_x2s);
    double x2s_cvx_hull_area = polygonArea(x2s_convex_hull);
    double e2_weff = 2. * sqrt(x2s_cvx_hull_area / PI);

    double mean_e2_error_x = thresholded_x2s.col(0).mean() - emitter_xy(1,0);
    double mean_e2_error_y = thresholded_x2s.col(1).mean() - emitter_xy(1,1);

    x1s_convex_hull = loopify(x1s_convex_hull);
    x2s_convex_hull = loopify(x2s_convex_hull);
   
    savePoints("tem-localization/x1s_convex_hull.csv", x1s_convex_hull);
    savePoints("tem-localization/x2s_convex_hull.csv", x2s_convex_hull);

    savePoints("tem-localization/x1s.csv", x1s);
    savePoints("tem-localization/x2s.csv", x2s);

    printf("> %s\n", (chi_2_method == NORMALIZE ? "normalized chi2" : "non-normalized chi2"));
    printf("> fit quality emitter 1: %0.4f, %0.4f, %0.4f\n", fitQuality(x1s), e1_weff, sqrt(mean_e1_error_x * mean_e1_error_x + mean_e1_error_y * mean_e1_error_y));
    printf("> fit quality emitter 2: %0.4f, %0.4f, %0.4f\n", fitQuality(x2s), e2_weff, sqrt(mean_e2_error_x * mean_e2_error_x + mean_e2_error_y * mean_e2_error_y));
    printf("> fit quality overall: %0.4f, %0.4f\n", 0.5 * (fitQuality(x1s) + fitQuality(x2s)), 0.5 * (e1_weff + e2_weff));
    printf(
        "$%0.4f$ & $%0.4f$ & $%0.4f$ & $%0.4f$ & $%0.4f$ & $%0.4f$\n",
        fitQuality(x1s), e1_weff, sqrt(mean_e1_error_x * mean_e1_error_x + mean_e1_error_y * mean_e1_error_y),
        fitQuality(x2s), e2_weff, sqrt(mean_e2_error_x * mean_e2_error_x + mean_e2_error_y * mean_e2_error_y)
    );

    return 0;
}