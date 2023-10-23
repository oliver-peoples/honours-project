#ifndef __ADIRS_FAST_GP_H__
#define __ADIRS_FAST_GP_H__

#include <complex>
#include <cstdio>
#include <math.h>
#include "utils.h"
#include <math.h>
#include <numeric>

#ifdef __NVCC__
#else
    #define __HOST_DEVICE__
#endif

#include <Eigen/Dense>

//=================================================================================================
// declare the polynomials
//=================================================================================================

// gauss-hermite

template <int order>
inline __HOST_DEVICE__ double fastGH(double x);

template <> inline __HOST_DEVICE__ double fastGH<0>(double x)
{
    return 1.;
}

template <> inline __HOST_DEVICE__ double fastGH<1>(double x)
{
    return 2. * x;
}

template <> inline __HOST_DEVICE__ double fastGH<2>(double x)
{
    return 4. * (x * x) - 2.;
}

template <> inline __HOST_DEVICE__ double fastGH<3>(double x)
{
    return 8. * (x * x * x) - 12. * x;
}

// gauss-laguerre

template <int order>
inline __HOST_DEVICE__ double fastGL(int index, double x);

template <> inline __HOST_DEVICE__ double fastGL<0>(int index, double x)
{
    return 1;
}

template <> inline __HOST_DEVICE__ double fastGL<1>(int index, double x)
{
    return -x + (double)index - 1;
}

template <> inline __HOST_DEVICE__ double fastGL<2>(int index, double x)
{
    // return 0.5 * ((x * x) - (4 * x) + 2);
    return 0.5 * x * x - x * double(index + 2) + 0.5 * double((index+1) * (index+2));
}

template <> inline __HOST_DEVICE__ double fastGL<3>(int index, double x)
{
    static constexpr double _1_6 = 1./6.;

    double xx = x * x;

    return _1_6 * (-(xx * x) + (9 * xx) - (18 * x) + 6);
}

// \begin{table}[h]
//     \centering
//     \caption{Associated Laguerre Polynomials for Substitution into GH TEM General Equation}
//     \label{tab:hermite-polynomials}
//     \begin{tabular}{c c}
//         \toprule
//         $p$ & \bf{Laguerre Polynomial}\\
//         \midrule
//         $0$ & $L_0^{l}(u) = 1$ \\
//         $1$ & $L_1^{l}(u) = -u+l+1$ \\
//         $2$ & $L_2^{l}(u) = \frac{x^2}{2} - (l + 2)x + \frac{(l+1)(l+2)}{2} - 2$\\
//         $3$ & $L_3^{l}(u) = \frac{-x^3}{6} + \frac{(l+3)x^2}{2} -\frac{(l+2)(l+3)x}{2} +\frac{(l+1)(l+2)(l+3)}{6}$\\
//         \bottomrule
//     \end{tabular}
// \end{table}

// template <int p, int l>
// struct FastGL
// {
//     double poly(double u)
//     {
//         if constexpr ( p == 0 )
//         {
//             return 1;
//         }
//         else if constexpr ( p == 1 )
//         {
//             return -u + (double)l + 1;
//         }
//         else if constexpr ( p == 2)
//         {
//             return 0.5 * u * u - u * double(l + 2) + 0.5 * double((l+1) * (l+2));
//         }
//         else if constexpr ( p == 3 )
//         {

//         }
//     }
// };

//=================================================================================================
// declare the illumination functions
//=================================================================================================

// template <int order_m, int order_n, double w, double I_0>
// inline __HOST_DEVICE__ double fastTEM_GH(Eigen::Vector2d xy)
// {
//     static constexpr double ww = w * w;

//     double x_comp = fastGH<order_m>((SQRT2 / 2) * xy.x()) * exp(-(xy.x() * xy.x()) / (ww));
//     double y_comp = fastGH<order_n>((SQRT2 / 2) * xy.y()) * exp(-(xy.y() * xy.y()) / (ww));

//     return I_0 * ww * (x_comp * x_comp) * (y_comp * y_comp);
// }

template <int order_m, int order_n>
inline __HOST_DEVICE__ double fastTEM_GH(Eigen::Vector2d xy, double w, double I_0, Eigen::Vector2d tx, double r)
{
    Eigen::Matrix2d r_mat {
        { cos(-r),-sin(-r) },
        { sin(-r),cos(-r) }
    };

    xy = r_mat * xy;

    xy -= tx;

    static const double ww = w * w;

    double x_comp = fastGH<order_m>((SQRT2 / w) * xy.x());
    double y_comp = fastGH<order_n>((SQRT2 / w) * xy.y());

    x_comp *= exp(-(xy.x() * xy.x()) / (ww));
    y_comp *= exp(-(xy.y() * xy.y()) / (ww));

    return I_0 * ww * (x_comp * x_comp) * (y_comp * y_comp);
}

template <int p, int l>
inline __HOST_DEVICE__ double fastTEM_GL(Eigen::Vector3d xyz, const double wl, const double w0, const double wn)
{
    constexpr double p_fact = 2. * factorial(p);
    constexpr double pi_pl_fact = PI * factorial(p+l);

    const double norm_factor = sqrt(p_fact / pi_pl_fact);

    double psi = atan2(xyz.y(), xyz.x());

    double r = Eigen::Vector2d(xyz.x(),xyz.y()).norm();

    double z = xyz.z();

    double z_r = PI * w0 * w0 + 1. / wl;

    double z_z_r = z / z_r;

    double w_z = w0 * sqrt(1 + z_z_r * z_z_r);

    double inv_2r = 0.5 * z / (z*z + z_r*z_r);

    double gouy_phase_shift = double(l + p + 1) * atan(z_z_r);

    double comp_1 = pow(SQRT2 * r / w_z, l);
    double comp_2 = exp(-(r*r) / (w_z * w_z));
    double comp_3 = std::assoc_laguerre(p, l, 2 * (r*r) / (w_z*w_z));
    std::complex<double> comp_4 = exp(std::complex<double>(-1. * wn * r*r * inv_2r,0));
    std::complex<double> comp_5 = exp(std::complex<double>(-1. * (double)l * psi,0));
    std::complex<double> comp_6 = exp(std::complex<double>(1. * gouy_phase_shift,0));

    std::complex<double> e_field = comp_1 * comp_2 * comp_3 * comp_4 * comp_5 * comp_6;

    double intensity = pow(e_field.real(), 2.0);

    return intensity;
}

struct Beam
{
    double (*tem)(Eigen::Vector3d xyz,const double wl, const double w0, const double wn);

    double pitch_deg = 0;
    double pivot_deg = 0;
    double roll_deg = 0;
    double z_offset = 0;

    Eigen::Vector3d mapTo(Eigen::Vector3d xyz)
    {
        double pitch_rad_2 = 0.5 * PI * pitch_deg / 180.;
        double pivot_rad_2 = 0.5 * PI * pivot_deg / 180.;
        double roll_rad_2 = 0.5 * PI * roll_deg / 180.;
        double sin_roll_rad_2 = sin(roll_rad_2);


        Eigen::Quaterniond pitch_rotor = {
            cos(pitch_rad_2),
            0,
            sin(pitch_rad_2),
            0
        };

        Eigen::Quaterniond pivot_rotor = {
            cos(pivot_rad_2),
            0,
            0,
            sin(pivot_rad_2)
        };

        xyz = (pitch_rotor * Eigen::Quaterniond(0,xyz.x(),xyz.y(),xyz.z()) * pitch_rotor.conjugate()).vec();
        xyz = (pivot_rotor * Eigen::Quaterniond(0,xyz.x(),xyz.y(),xyz.z()) * pivot_rotor.conjugate()).vec();

        Eigen::Quaterniond roll_rotor = {
            cos(roll_rad_2),
            xyz.x() * sin_roll_rad_2,
            xyz.y() * sin_roll_rad_2,
            xyz.z() * sin_roll_rad_2
        };

        xyz = (roll_rotor * Eigen::Quaterniond(0,xyz.x(),xyz.y(),xyz.z()) * roll_rotor.conjugate()).vec();

        xyz.z() -= z_offset;

        return xyz;
    }
};

#endif /* __ADIRS_FAST_GP_H__ */
