#ifndef __MULTICORE_EXPERIMENTS_H__
#define __MULTICORE_EXPERIMENTS_H__

#include "utils.h"

//=====================================================================================================================
// infinite time
//=====================================================================================================================

inline ArrX2d multicoreMeasureInfTime(
    ArrX3d& core_locations,
    Eigen::VectorXi& g2_capable_idx,
    Eigen::Array<double,2,3> emitter_xy,
    Eigen::Array<double,2,1> emitter_brightness
)
{
    Eigen::ArrayX2d distances(core_locations.rows(), 2);

    for (int row_num = 0; row_num < core_locations.rows(); row_num++)
    {
        Eigen::Vector3d e_1_diff_vec = core_locations.row(row_num) - emitter_xy.row(0);
        Eigen::Vector3d e_2_diff_vec = core_locations.row(row_num) - emitter_xy.row(1);

        distances(row_num,0) = e_1_diff_vec.norm();
        distances(row_num,1) = e_2_diff_vec.norm();
    }

    Eigen::ArrayX2d powers = Eigen::exp(-(distances * distances / 2)/(2 * (1 * 1)));

    powers.col(0) *= emitter_brightness(0);
    powers.col(1) *= emitter_brightness(1);

    Eigen::ArrayX2d multicore_measure = powers;

    multicore_measure.col(0) = (powers.col(0) + powers.col(1)) / (emitter_brightness(0) + emitter_brightness(1));

    multicore_measure.col(1) = 0;

    Eigen::Array<double,Eigen::Dynamic,1> alpha = powers(g2_capable_idx,0) / powers(g2_capable_idx,1);

    Eigen::Array<double,Eigen::Dynamic,1> alpha_p1 = alpha + 1.;

    multicore_measure(g2_capable_idx,1) = (2. * alpha) / (alpha_p1 * alpha_p1);

    return multicore_measure;
}

struct MulticoreDataInfTime
{
    ArrX3d& core_locations;
    ArrX2d& multicore_measure_noisy;
    Eigen::VectorXi& g2_capable_idx;
    CHI2_METHOD chi_2_method = NORMALIZE;
};

inline double multicoreInfTimeChi2(
    const Eigen::VectorXd& vals_inp,
    Eigen::VectorXd* grad_out,
    void* opt_data
)
{
    MulticoreDataInfTime* multicore_data_ptr = (MulticoreDataInfTime*)opt_data;
    
    ArrX3d& core_locations = multicore_data_ptr->core_locations;
    ArrX2d& multicore_measure_noisy = multicore_data_ptr->multicore_measure_noisy;
    Eigen::VectorXi& g2_capable_idx = multicore_data_ptr->g2_capable_idx;

    Eigen::Array<double,2,3> emitter_xy {
        { vals_inp(0),vals_inp(1),0 },
        { vals_inp(2),vals_inp(3),0 }
    };

    Eigen::Array<double,2,1> emitter_brightness {
        1.,vals_inp(4)
    };

    Eigen::ArrayX2d distances(core_locations.rows(), 2);

    for (int row_num = 0; row_num < core_locations.rows(); row_num++)
    {
        Eigen::Vector3d e_1_diff_vec = core_locations.row(row_num) - emitter_xy.row(0);
        Eigen::Vector3d e_2_diff_vec = core_locations.row(row_num) - emitter_xy.row(1);

        distances(row_num,0) = e_1_diff_vec.norm();
        distances(row_num,1) = e_2_diff_vec.norm();
    }

    Eigen::ArrayX2d powers = Eigen::exp(-(distances * distances / 2)/(2 * (1 * 1)));

    powers.col(0) *= emitter_brightness(0);
    powers.col(1) *= emitter_brightness(1);

    Eigen::ArrayX2d multicore_measure = powers;

    multicore_measure.col(0) = (powers.col(0) + powers.col(1)) / (emitter_brightness(0) + emitter_brightness(1));

    multicore_measure.col(1) = 0;

    Eigen::Array<double,Eigen::Dynamic,1> alpha = powers(g2_capable_idx,0) / powers(g2_capable_idx,1);

    Eigen::Array<double,Eigen::Dynamic,1> alpha_p1 = alpha + 1.;

    multicore_measure(g2_capable_idx,1) = (2. * alpha) / (alpha_p1 * alpha_p1);

    ArrX2d chi2 = (multicore_measure - multicore_measure_noisy).pow(2);

    // chi2 *= chi2;

    if ( multicore_data_ptr->chi_2_method == NORMALIZE )
    {
        chi2.col(0) /= multicore_measure_noisy.col(0);
        chi2(g2_capable_idx,1) /= multicore_measure_noisy(g2_capable_idx,1);
    }

    return chi2.col(0).sum() + chi2.col(1).sum();
}


//=====================================================================================================================
// infinite time
//=====================================================================================================================

inline ArrX2d multicoreMeasureFiniteTime(
    ArrX3d& core_locations,
    Eigen::VectorXi& g2_capable_idx,
    Eigen::Array<double,2,3> emitter_xy,
    Eigen::Array<double,2,1> emitter_brightness
)
{
    Eigen::ArrayX2d distances(core_locations.rows(), 2);

    for (int row_num = 0; row_num < core_locations.rows(); row_num++)
    {
        Eigen::Vector3d e_1_diff_vec = core_locations.row(row_num) - emitter_xy.row(0);
        Eigen::Vector3d e_2_diff_vec = core_locations.row(row_num) - emitter_xy.row(1);

        distances(row_num,0) = e_1_diff_vec.norm();
        distances(row_num,1) = e_2_diff_vec.norm();
    }

    Eigen::ArrayX2d powers = Eigen::exp(-(distances * distances / 2)/(2 * (1 * 1)));

    powers.col(0) *= emitter_brightness(0);
    powers.col(1) *= emitter_brightness(1);

    Eigen::ArrayX2d multicore_measure = powers;

    multicore_measure.col(0) = (powers.col(0) + powers.col(1)) / (emitter_brightness(0) + emitter_brightness(1));

    multicore_measure.col(1) = 0;

    Eigen::Array<double,Eigen::Dynamic,1> alpha = powers(g2_capable_idx,0) / powers(g2_capable_idx,1);

    Eigen::Array<double,Eigen::Dynamic,1> alpha_p1 = alpha + 1.;

    multicore_measure(g2_capable_idx,1) = (2. * alpha) / (alpha_p1 * alpha_p1);

    return multicore_measure;
}

struct MulticoreDataFiniteTime
{
    ArrX3d& core_locations;
    ArrX2d& multicore_measure_noisy;
    Eigen::VectorXi& g2_capable_idx;
    double exposure_time;
    CHI2_METHOD chi_2_method = NORMALIZE;
};

inline double multicoreFiniteTimeChi2(
    const Eigen::VectorXd& vals_inp,
    Eigen::VectorXd* grad_out,
    void* opt_data
)
{
    MulticoreDataFiniteTime* multicore_data_ptr = (MulticoreDataFiniteTime*)opt_data;
    
    ArrX3d& core_locations = multicore_data_ptr->core_locations;
    ArrX2d& multicore_measure_noisy = multicore_data_ptr->multicore_measure_noisy;
    Eigen::VectorXi& g2_capable_idx = multicore_data_ptr->g2_capable_idx;

    Eigen::Array<double,2,3> emitter_xy {
        { vals_inp(0),vals_inp(1),0 },
        { vals_inp(2),vals_inp(3),0 }
    };

    Eigen::Array<double,2,1> emitter_brightness {
        1.,vals_inp(4)
    };

    Eigen::ArrayX2d distances(core_locations.rows(), 2);

    for (int row_num = 0; row_num < core_locations.rows(); row_num++)
    {
        Eigen::Vector3d e_1_diff_vec = core_locations.row(row_num) - emitter_xy.row(0);
        Eigen::Vector3d e_2_diff_vec = core_locations.row(row_num) - emitter_xy.row(1);

        distances(row_num,0) = e_1_diff_vec.norm();
        distances(row_num,1) = e_2_diff_vec.norm();
    }

    Eigen::ArrayX2d powers = Eigen::exp(-(distances * distances / 2)/(2 * (1 * 1)));

    powers.col(0) *= emitter_brightness(0);
    powers.col(1) *= emitter_brightness(1);

    Eigen::ArrayX2d multicore_measure = powers;

    multicore_measure.col(0) = (powers.col(0) + powers.col(1)) / (emitter_brightness(0) + emitter_brightness(1));

    multicore_measure.col(1) = 0;

    Eigen::Array<double,Eigen::Dynamic,1> alpha = powers(g2_capable_idx,0) / powers(g2_capable_idx,1);

    Eigen::Array<double,Eigen::Dynamic,1> alpha_p1 = alpha + 1.;

    multicore_measure(g2_capable_idx,1) = (2. * alpha) / (alpha_p1 * alpha_p1);

    ArrX2d chi2 = (multicore_measure - multicore_measure_noisy).pow(2);

    // chi2 *= chi2;

    if ( multicore_data_ptr->chi_2_method == NORMALIZE )
    {
        chi2.col(0) /= multicore_measure_noisy.col(0);
        chi2(g2_capable_idx,1) /= multicore_measure_noisy(g2_capable_idx,1);
    }

    return chi2.col(0).sum() + chi2.col(1).sum();
}

#endif /* __MULTICORE_EXPERIMENTS_H__ */
