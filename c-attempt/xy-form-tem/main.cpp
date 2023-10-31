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

constexpr double wl = 633e-9;
constexpr double w0 = 633e-9;
constexpr double wn = 2 * PI / wl;

constexpr double detector_w = 1.;
constexpr int TRIALS_PER_CONFIG = 10000;

constexpr CHI2_METHOD chi_2_method = WORBOY;

constexpr double pm_x_w0 = 1.5;
constexpr double pm_y_w0 = 1.5;
double pm_z_w0 = (1 / w0) * 3 * PI * w0*w0 * 1. / wl;

constexpr long x_samples = 1000;
constexpr long y_samples = 1000;
constexpr long z_samples = 1;

constexpr long num_samples = x_samples * y_samples * z_samples;

// std::vector<Beam> beams = {
//     {
//         fastTEM_GH<0,0>,
//         w0,
//         { 0.0 * w0,1.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GH<0,1>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GH<1,0>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GH<1,1>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GH<1,2>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GH<2,1>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GH<2,2>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     }
// };

// std::vector<Beam> beams = {
//     {
//         fastTEM_GL<0,0>,
//         w0,
//         { 0.0 * w0,1.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GL<0,1>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GL<1,0>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GL<1,1>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GL<1,2>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GL<2,1>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     },
//     {
//         fastTEM_GL<2,2>,
//         w0,
//         { 0.0 * w0,0.0 * w0 },
//         0,
//         0,
//         0
//     }
// };


std::vector<Beam> beams = {
    {
        fastTEM_GL<0,0>,
        w0,
        { 0.0 * w0,1.0 * w0 },
        0,
        0,
        0
    },
    {
        fastTEM_GL<1,2>,
        w0,
        { 0.0 * w0,0.0 * w0 },
        45,
        0,
        0
    },
    {
        fastTEM_GL<1,2>,
        w0,
        { 0.0 * w0,0.0 * w0 },
        45,
        10,
        5
    },
    {
        fastTEM_GL<1,2>,
        w0,
        { 0.0 * w0,0.0 * w0 },
        45,
        20,
        10
    },
    {
        fastTEM_GL<1,2>,
        w0,
        { 0.0 * w0,0.0 * w0 },
        45,
        30,
        15
    },
    {
        fastTEM_GL<1,2>,
        w0,
        { 0.0 * w0,0.0 * w0 },
        45,
        40,
        20.
    },
    {
        fastTEM_GL<1,2>,
        w0,
        { 0.0 * w0,0.0 * w0 },
        45,
        50,
        25.
    }
    // {
    //     fastTEM_GH<1,2>,
    //     w0,
    //     { -0.1 * w0,0.0 * w0 },
    //     45,
    //     0,
    //     0
    // },
    // {
    //     fastTEM_GH<1,2>,
    //     w0,
    //     { 0.0 * w0,-0.1 * w0 },
    //     45,
    //     10,
    //     5
    // },
    // {
    //     fastTEM_GH<1,2>,
    //     w0,
    //     { 0.1 * w0,0.0 * w0 },
    //     45,
    //     20,
    //     10
    // },
    // {
    //     fastTEM_GH<1,2>,
    //     w0,
    //     { 0.0 * w0,0.1 * w0 },
    //     45,
    //     30,
    //     15
    // },
    // {
    //     fastTEM_GH<1,2>,
    //     w0,
    //     { 0.0 * w0,0.0 * w0 },
    //     45,
    //     40,
    //     20.
    // },
    // {
    //     fastTEM_GH<1,2>,
    //     w0,
    //     { 0.0 * w0,0.0 * w0 },
    //     45,
    //     50,
    //     25.
    // }
};

int main()
{
    ArrX3d beam_centers = ArrX3d(beams.size(),3);

    for (int beam_idx = 0; beam_idx < beams.size(); beam_idx++)
    {
        beam_centers.row(beam_idx) = (1. / w0) * Eigen::Vector3d{ beams[beam_idx].center.x(),beams[beam_idx].center.y(),0. };
    }

    savePoints("xy-form-tem/beam_centers.csv", beam_centers);
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

    savePoints("xy-form-tem/emitter_xy.csv", emitter_xy);

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

    saveMeasurements("xy-form-tem/g1_g2_measurements.csv", tem_measure);

    double variab = 0.0;
    double variab_2 = 1.;

    if constexpr (z_samples == 1) { pm_z_w0 = 0; }

    FILE* meta_out = fopen("xy-form-tem/meta.out", "w");

    fprintf(meta_out, "%li,%li,%li\n", x_samples, y_samples, z_samples);
    fprintf(meta_out, "%f,%f,%f\n", -pm_x_w0, -pm_x_w0, -pm_x_w0);
    fprintf(meta_out, "%f,%f,%f\n", pm_x_w0, pm_x_w0, pm_x_w0);

    fclose(meta_out);
    
    double* chi2_array = (double*)malloc(sizeof(double) * num_samples);

    auto start = high_resolution_clock::now();

    // omp_set_num_threads(1);
    #pragma omp parallel
    for (long linear_idx = omp_get_thread_num(); linear_idx < num_samples; linear_idx += omp_get_num_threads())
    {
        long x_idx = linear_idx % x_samples;
        long y_idx = (linear_idx % (x_samples * y_samples)) / x_samples;
        long z_idx = linear_idx / (x_samples * y_samples);

        double x_space = -pm_x_w0 + 2*pm_x_w0 * (double)x_idx / double(x_samples - 1);
        double y_space = -pm_y_w0 + 2*pm_y_w0 * (double)y_idx / double(y_samples - 1);
        double z_space;

        ArrX2d tem_measure_noisy = tem_measure;

        tem_measure_noisy *= (1 + variab * Eigen::Array<double,Eigen::Dynamic,2>::Random(tem_measure.rows(),2));

        Eigen::Array<double,2,3> emitter_xy_guess {
            { -0.6300,-0.1276,0 },
            { x_space,y_space,0 }
        };

        ArrX2d tem_measure = temMeasureInfTime(
            beams,
            emitter_xy_guess,
            emitter_brightness,
            wl,
            w0,
            wn,
            detector_w
        );

        ArrX2d chi2 = (tem_measure - tem_measure_noisy).pow(2);

        if ( chi_2_method == NORMALIZE )
        {
            chi2.col(0) /= tem_measure_noisy.col(0);
            chi2.col(1) /= tem_measure_noisy.col(1);
        }

        chi2_array[linear_idx] = chi2.col(0).sum() + chi2.col(1).sum();

        // Eigen::VectorXd xx = Eigen::Array<double,5,1>::Random(5,1);
        // // Eigen::VectorXd xx = Eigen::Array<double,5,1>{ emitter_xy(0,0),emitter_xy(0,1),emitter_xy(1,0),emitter_xy(1,1),emitter_brightness[1] };
        // // xx.array() *= (1 + variab_2 * Eigen::Array<double,5,1>::Random());

        // xx(4) = 0.5;

        // TemDataInfTime tem_data = {
        //     beams, tem_measure_noisy, wl, w0, wn, detector_w, NORMALIZE
        // };

        // bool success = optim::nm(xx, temInfTimeChi2, (void*)&tem_data);

        // if ( xx(4) < 1 )
        // {
        //     x1s(cts,1 - 1) = xx(1 - 1);
        //     x1s(cts,2 - 1) = xx(2 - 1);
        //     x2s(cts,1 - 1) = xx(3 - 1);
        //     x2s(cts,2 - 1) = xx(4 - 1);
        //     p02s(cts) = xx(5 - 1);
        // }
        // else
        // {
        //     x1s(cts,1 - 1) = xx(3 - 1);
        //     x1s(cts,2 - 1) = xx(4 - 1);
        //     x2s(cts,1 - 1) = xx(1 - 1);
        //     x2s(cts,2 - 1) = xx(2 - 1);
        //     p02s(cts) = 1/xx(5 - 1);
        // }

        // ArrX2d tem_measure_noisy = tem_measure * (1 + ArrX2d::Random())
    }

    FILE* data_out = fopen("xy-form-tem/chi_2.bin", "wb");

    fwrite((void*)chi2_array, sizeof(double), num_samples, data_out);

    fclose(data_out);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << duration.count() << ": " << (double)duration.count() / (double)TRIALS_PER_CONFIG << std::endl;

    return 0;
}