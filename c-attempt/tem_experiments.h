#ifndef __TEM_EXPERIMENTS_H__
#define __TEM_EXPERIMENTS_H__

#include <Eigen/Dense>
#include "utils.h"
#include "fast_gp.h"

inline ArrX2d temMeasureInfTime(
    std::vector<Beam>& beams,
    Eigen::Array<double,2,3> emitter_xy,
    Eigen::Array<double,2,1> emitter_brightness,
    double wl,
    double w0,
    double wn
)
{
    emitter_xy *= w0;

    ArrX3d beam_centers = ArrX3d(beams.size(),3);

    for (int beam_idx = 0; beam_idx < beams.size(); beam_idx++)
    {
        beam_centers.row(beam_idx) = Eigen::Vector3d{ beams[beam_idx].center.x(),beams[beam_idx].center.y(),0. };
    }

    Eigen::ArrayX2d distances(beam_centers.rows(), 2);

    Eigen::ArrayX2d field_intensity = distances;

    for (int row_num = 0; row_num < beam_centers.rows(); row_num++)
    {
        Eigen::Vector3d e_1_diff_vec = beam_centers.row(row_num) - emitter_xy.row(0);
        Eigen::Vector3d e_2_diff_vec = beam_centers.row(row_num) - emitter_xy.row(1);

        distances(row_num,0) = e_1_diff_vec.norm();
        distances(row_num,1) = e_2_diff_vec.norm();

        field_intensity(row_num,0) = beams[row_num].tem(
            beams[row_num].mapTo(emitter_xy.row(0)),
            wl,
            w0,
            wn
        );

        field_intensity(row_num,1) = beams[row_num].tem(
            beams[row_num].mapTo(emitter_xy.row(1)),
            wl,
            w0,
            wn
        );
    }

    // std:: cout << field_intensity << std::endl;

    Eigen::ArrayX2d powers = field_intensity * Eigen::exp(-(distances * distances / 2)/(2 * (1 * 1)));

    powers.col(0) *= emitter_brightness(0);
    powers.col(1) *= emitter_brightness(1);

    Eigen::ArrayX2d tem_measure = powers;

    tem_measure.col(0) = (powers.col(0) + powers.col(1)) / (emitter_brightness(0) + emitter_brightness(1));

    tem_measure.col(1) = 0;

    Eigen::Array<double,Eigen::Dynamic,1> alpha = powers.col(0) / powers.col(1);

    Eigen::Array<double,Eigen::Dynamic,1> alpha_p1 = alpha + 1.;

    tem_measure.col(1) = (2. * alpha) / (alpha_p1 * alpha_p1);

    return tem_measure;
}

struct TemDataInfTime
{
    std::vector<Beam>& beams;
    ArrX2d tem_measure_noisy;
    double wl;
    double w0;
    double wn;
    CHI2_METHOD chi_2_method = NORMALIZE;
};

inline double temInfTimeChi2(
    const Eigen::VectorXd& vals_inp,
    Eigen::VectorXd* grad_out,
    void* opt_data
)
{
    TemDataInfTime* tem_data_ptr = (TemDataInfTime*)opt_data;

    std::vector<Beam>& beams = tem_data_ptr->beams;

    ArrX2d& tem_measure_noisy = tem_data_ptr->tem_measure_noisy;

    double wl = tem_data_ptr->wl;
    double w0 = tem_data_ptr->w0;
    double wn = tem_data_ptr->wn;

    Eigen::Array<double,2,3> emitter_xy {
        { vals_inp(0),vals_inp(1),0 },
        { vals_inp(2),vals_inp(3),0 }
    };

    Eigen::Array<double,2,1> emitter_brightness {
        1.,vals_inp(4)
    };

    emitter_xy *= w0;

    ArrX3d beam_centers = ArrX3d(beams.size(),3);

    for (int beam_idx = 0; beam_idx < beams.size(); beam_idx++)
    {
        beam_centers.row(beam_idx) = Eigen::Vector3d{ beams[beam_idx].center.x(),beams[beam_idx].center.y(),0. };
    }

    Eigen::ArrayX2d distances(beam_centers.rows(), 2);

    Eigen::ArrayX2d field_intensity = distances;

    for (int row_num = 0; row_num < beam_centers.rows(); row_num++)
    {
        Eigen::Vector3d e_1_diff_vec = beam_centers.row(row_num) - emitter_xy.row(0);
        Eigen::Vector3d e_2_diff_vec = beam_centers.row(row_num) - emitter_xy.row(1);

        distances(row_num,0) = e_1_diff_vec.norm();
        distances(row_num,1) = e_2_diff_vec.norm();

        field_intensity(row_num,0) = beams[row_num].tem(
            beams[row_num].mapTo(emitter_xy.row(0)),
            wl,
            w0,
            wn
        );

        field_intensity(row_num,1) = beams[row_num].tem(
            beams[row_num].mapTo(emitter_xy.row(1)),
            wl,
            w0,
            wn
        );
    }

    // std:: cout << field_intensity << std::endl;

    Eigen::ArrayX2d powers = field_intensity * Eigen::exp(-(distances * distances / 2)/(2 * (1 * 1)));

    powers.col(0) *= emitter_brightness(0);
    powers.col(1) *= emitter_brightness(1);

    Eigen::ArrayX2d tem_measure = powers;

    tem_measure.col(0) = (powers.col(0) + powers.col(1)) / (emitter_brightness(0) + emitter_brightness(1));

    tem_measure.col(1) = 0;

    Eigen::Array<double,Eigen::Dynamic,1> alpha = powers.col(0) / powers.col(1);

    Eigen::Array<double,Eigen::Dynamic,1> alpha_p1 = alpha + 1.;

    tem_measure.col(1) = (2. * alpha) / (alpha_p1 * alpha_p1);

    ArrX2d chi2 = (tem_measure - tem_measure_noisy).pow(2);

    return chi2.col(0).sum() + chi2.col(1).sum();
}

#endif /* __TEM_EXPERIMENTS_H__ */
