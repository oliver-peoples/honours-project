#include "fast_gp.h"
#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <omp.h>
// #include <math.h>
#include <cmath>
#include "utils.h"

#define USE_OMP 1

// #define OPTIM_ENABLE_EIGEN_WRAPPERS

constexpr double PM_X = 1.5;
constexpr double PM_Y = 1.5;

constexpr int NUM_X = 2000;
constexpr int NUM_Y = 2000;

constexpr int M = 3;
constexpr int N = 3;

constexpr int NUM_SAMPLES = NUM_X * NUM_Y;

constexpr int m_fact = factorial(M);
constexpr int n_fact = factorial(N);

int main()
{
    double* i_fn = (double*)malloc(NUM_X * NUM_Y * sizeof(double));

    double norm = sqrt(2/(3.14159 * double(n_fact * m_fact))) * pow(2., -double(N+M)/2);

    double (*tem)(Eigen::Vector2d xy, double w, double I_0, Eigen::Vector2d tx, double r) = &fastTEM_GH<M,N>;

    printf("> Calculating TEM mode...\n");

    #if USE_OMP
    #pragma omp parallel
    for (int linear_idx = omp_get_thread_num(); linear_idx < NUM_SAMPLES; linear_idx += omp_get_num_threads())
    #else
    for (int linear_idx = 0; linear_idx < NUM_SAMPLES; linear_idx++)
    #endif
    {
        int x_idx = linear_idx % NUM_X;
        int y_idx = linear_idx / NUM_X;

        double x_val = (double)x_idx / double(NUM_X - 1);
        double y_val = (double)y_idx / double(NUM_Y - 1);

        x_val = -1. + 2. * x_val;
        y_val = 1. - 2. * y_val;

        x_val *= PM_X;
        y_val *= PM_Y;

        // Eigen::Vector2d xy = { x_val,y_val };

        // double x_comp = (SQRT2 / 1.) * xy.x();
        // double y_comp = (SQRT2 / 1.) * xy.y();

        // x_comp = 4. * (x_comp * x_comp) - 2.;
        // y_comp = 4. * (y_comp * y_comp) - 2.;

        // x_comp *= exp(-(xy.x() * xy.x()) / (1.));
        // y_comp *= exp(-(xy.y() * xy.y()) / (1.));

        // i_fn[linear_idx] = 1. * 1. * (x_comp * x_comp) * (y_comp * y_comp);

        i_fn[linear_idx] = norm * tem(
            { x_val,y_val },
            0.5,
            1.,
            { 0.5,0.5 },
            30. * PI / 180.
        );
    }

    printf("> Done.\n");

    FILE* i_fn_data = fopen("i_fn_out.bin", "wb");

    fwrite(&NUM_X, sizeof(int), 1, i_fn_data);
    fwrite(&NUM_Y, sizeof(int), 1, i_fn_data);
    fwrite(&PM_X, sizeof(double), 1, i_fn_data);
    fwrite(&PM_Y, sizeof(double), 1, i_fn_data);
    fwrite(&M, sizeof(int), 1, i_fn_data);
    fwrite(&N, sizeof(int), 1, i_fn_data);
    
    fwrite(&i_fn[0], sizeof(double), NUM_SAMPLES, i_fn_data);

    fclose(i_fn_data);

    system("python render_tem.py");

    return 0;
}