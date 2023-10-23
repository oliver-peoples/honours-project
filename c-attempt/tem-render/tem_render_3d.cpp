#include <Eigen/Dense>
#include <iostream>
#include <omp.h>
#include <vector>

#define OPTIM_ENABLE_EIGEN_WRAPPERS

// #include <optim.hpp>

#include "../fast_gp.h"
#include "../utils.h"

constexpr double wl = 633e-9;
constexpr double w0 = 1e-9;
constexpr double wn = 2 * PI / wl;

constexpr double pm_x_w0 = 2.;
constexpr double pm_y_w0 = 2.;
constexpr double pm_z_w0 = 10.;

constexpr long x_samples = 300;
constexpr long y_samples = 300;
constexpr long z_samples = 1500;

constexpr long num_samples = x_samples * y_samples * z_samples;

Beam beam = {
    fastTEM_GL<2, 4>,
    35
};

int main()
{

    FILE* meta_out = fopen("tem-render/meta.out", "w");

    fprintf(meta_out, "%li,%li,%li\n", x_samples, y_samples, z_samples);
    fprintf(meta_out, "%f,%f,%f\n", -pm_x_w0, -pm_x_w0, -pm_x_w0);
    fprintf(meta_out, "%f,%f,%f\n", pm_x_w0, pm_x_w0, pm_x_w0);

    fclose(meta_out);
    
    double* tem_intensity = (double*)malloc(sizeof(double) * num_samples);

    #pragma omp parallel
    for (long linear_idx = omp_get_thread_num(); linear_idx < num_samples; linear_idx += omp_get_num_threads())
    {
        int x_idx = linear_idx % x_samples;
        int y_idx = linear_idx % (x_samples * y_samples) / x_samples;
        int z_idx = linear_idx / (x_samples * y_samples);

        double x_space = -pm_x_w0 + 2*pm_x_w0 * (double)x_idx / double(x_samples - 1);
        double y_space = -pm_y_w0 + 2*pm_y_w0 * (double)y_idx / double(y_samples - 1);
        double z_space = -pm_z_w0 + 2*pm_z_w0 * (double)z_idx / double(z_samples - 1);

        Eigen::Vector3d xyz = { x_space,y_space,z_space };
        xyz *= w0;

        xyz = beam.mapTo(xyz);

        double intensity = beam.tem(xyz, wl, w0, wn);

        tem_intensity[linear_idx] = intensity;
    }

    FILE* data_out = fopen("tem-render/tem_intensitys.bin", "wb");

    fwrite((void*)tem_intensity, sizeof(double), num_samples, data_out);

    fclose(data_out);

    return 0;
}

