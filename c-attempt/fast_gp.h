#ifndef __FAST_GP_H__
#define __FAST_GP_H__

#include <cstdio>

#ifdef __NVCC__
#else
    #define __HOST_DEVICE__
#endif

#include <Eigen/Dense>

constexpr double SQRT2 = 1.414213562373095048801688724209;
constexpr double PI = 3.14159265358979323846;

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
inline __HOST_DEVICE__ double fastGL(double x);

template <> inline __HOST_DEVICE__ double fastGL<0>(double x)
{
    return 1;
}

template <> inline __HOST_DEVICE__ double fastGL<1>(double x)
{
    return -x + 1;
}

template <> inline __HOST_DEVICE__ double fastGL<2>(double x)
{
    return 0.5 * ((x * x) - (4 * x) + 2);
}

template <> inline __HOST_DEVICE__ double fastGL<3>(double x)
{
    static constexpr double _1_6 = 1./6.;

    double xx = x * x;

    return _1_6 * (-(xx * x) + (9 * xx) - (18 * x) + 6);
}

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

#endif /* __FAST_GP_H__ */
