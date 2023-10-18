#ifndef __UTILS_H__
#define __UTILS_H__

#include <Eigen/Dense>

#include <algorithm>
#include <cstdlib>
#include <iostream>

constexpr double SQRT2 = 1.414213562373095048801688724209;
constexpr double PI = 3.14159265358979323846;

typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector2d Vec2d;
typedef Eigen::MatrixX3d MatX3d;
typedef Eigen::MatrixX2d MatX2d;
typedef Eigen::Matrix3d Mat3d;
typedef Eigen::ArrayX3d ArrX3d;
typedef Eigen::ArrayX2d ArrX2d;
typedef Eigen::Array3d Arr3d;

enum CHI2_METHOD
{
    NORMALIZE,
    WORBOY
};

const double CONF_FRAC = 1 - 1/sqrt(exp(1));
const double SQRT3 = sqrt(3.);

inline void createConcentricCores(ArrX3d& core_locations, int concentric_rings, double intercore_dist=1.)
{
    // H(n) = n^3 - (n-1)^3 = 3n(n-1)+1 = 3n^2 - 3n +1.

    int n = concentric_rings + 1;

    int num_cores = 3 * (n * n) - 3 * n + 1;

    core_locations = ArrX3d(num_cores, 3);

    core_locations.row(0) = Vec3d{ 0,0,0 };

    int idx_accumulator = 1;

    for(int ring_num = 1; ring_num < n; ring_num++)
    {
        int ring_points = (
            (3 * ((ring_num + 1) * (ring_num + 1)) - 3 * (ring_num + 1) + 1)
            -
            (3 * (ring_num * ring_num) - 3 * ring_num + 1)
        );

        int bridge_points = ((ring_points - 6) / 6);

        Eigen::Vector2d bl_vertex = { -0.5,-0.5 * SQRT3 };
    
        bl_vertex *= (double)ring_num;

        for (int vertex_point_num = 0; vertex_point_num < 6; vertex_point_num++)
        {
            Eigen::Matrix2d rotation_mat {
                { cos((double)vertex_point_num * PI / 3),-sin((double)vertex_point_num * PI / 3) },
                { sin((double)vertex_point_num * PI / 3),cos((double)vertex_point_num * PI / 3) }
            };

            Eigen::Vector2d vertex_point = rotation_mat * bl_vertex;

            core_locations.row(idx_accumulator) = Vec3d{ vertex_point.x(),vertex_point.y(),0 };
            core_locations.row(idx_accumulator) *= intercore_dist;

            idx_accumulator++;

            for (int non_vertex_point = 0; non_vertex_point < bridge_points; non_vertex_point++)
            {
                Eigen::Vector2d base_position = bl_vertex;

                base_position.x() += (double)ring_num * double(non_vertex_point + 1) * 1. / double(bridge_points + 1);

                Eigen::Vector2d side_point = rotation_mat * base_position;

                core_locations.row(idx_accumulator) = Vec3d{ side_point.x(),side_point.y(),0. };
                core_locations.row(idx_accumulator) *= intercore_dist;

                idx_accumulator++;
            }
        }
    }
}

inline void createWorboyCores(ArrX3d& core_locations, Eigen::VectorXi& g2_capable_idx)
{
    core_locations = ArrX3d(3,3);

    core_locations.row(0) = Vec3d{ 0.,1.,0  };
    core_locations.row(1) = Vec3d{ -sin(60*PI/180),-cos(60*PI/180),0 };
    core_locations.row(2) = Vec3d{ sin(60*PI/180),-cos(60*PI/180),0 };

    g2_capable_idx = Eigen::VectorXi(3);

    g2_capable_idx[0] = 0;
    g2_capable_idx[1] = 1;
    g2_capable_idx[2] = 2;

}

inline void savePoints(std::string file_name, ArrX3d& points)
{
    int num_rows = points.rows();

    FILE* f = fopen(file_name.c_str(), "w");

    fprintf(f, "pt_idx,x,y,z\n");

    for (int row_num = 0; row_num < num_rows; row_num++)
    {
        fprintf(
        f,
        "%i,%f,%f,%f",
            row_num,
            points(row_num,0),
            points(row_num,1),
            points(row_num,2)
        );

        if ( row_num < num_rows - 1 )
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

inline void savePoints(std::string file_name, ArrX2d& points)
{
    int num_rows = points.rows();

    FILE* f = fopen(file_name.c_str(), "w");

    fprintf(f, "pt_idx,x,y,z\n");

    for (int row_num = 0; row_num < num_rows; row_num++)
    {
        fprintf(
        f,
        "%i,%f,%f",
            row_num,
            points(row_num,0),
            points(row_num,1)
        );

        if ( row_num < num_rows - 1 )
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

inline void savePoints(std::string file_name, Eigen::Array<double,2,3>& points)
{
    int num_rows = points.rows();

    FILE* f = fopen(file_name.c_str(), "w");

    fprintf(f, "pt_idx,x,y\n");

    for (int row_num = 0; row_num < num_rows; row_num++)
    {
        fprintf(
        f,
        "%i,%f,%f,%f",
            row_num,
            points(row_num,0),
            points(row_num,1),
            points(row_num,2)
        );

        if ( row_num < num_rows - 1 )
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

inline void saveIndexes(std::string file_name, Eigen::VectorXi indexes)
{
    FILE* f = fopen(file_name.c_str(), "w");

    fprintf(f, "idx_idx,idx\n");

    int num_indexes = indexes.rows();

    for (int point_num = 0; point_num < num_indexes; point_num++)
    {
        fprintf(
        f,
        "%i,%i",
            point_num,
            indexes[point_num]
        );

        if ( point_num < num_indexes - 1 )
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

inline void saveMeasurements(std::string file_name, ArrX2d measurements)
{
    int num_rows = measurements.rows();

    FILE* f = fopen(file_name.c_str(), "w");

    fprintf(f, "core_idx,g1,g2\n");

    for (int row_num = 0; row_num < num_rows; row_num++)
    {
        fprintf(
        f,
        "%i,%f,%f",
            row_num,
            measurements(row_num,0),
            measurements(row_num,1)
        );

        if ( row_num < num_rows - 1 )
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

template <int rows, int cols>
inline void savePoints(std::string file_name, Eigen::Array<double,rows,cols> measurements)
{
    int num_rows = measurements.rows();
    int num_cols = measurements.cols();

    FILE* f = fopen(file_name.c_str(), "w");

    fprintf(f, "core_idx,g1,g2\n");

    for (int row_num = 0; row_num < num_rows; row_num++)
    {
        for (int col_num = 0; col_num < num_cols; col_num++)
        {
            fprintf(f,"%f", measurements(row_num,col_num));

            if ( col_num < num_cols - 1 )
            {
                fprintf(f, ",");
            }
        }

        if ( row_num < num_rows - 1 )
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

template <int rows, int cols>
inline void saveHeatmap(std::string file_name, Eigen::Array<double,rows,cols> measurements)
{
    int num_rows = measurements.rows();
    int num_cols = measurements.cols();

    FILE* f = fopen(file_name.c_str(), "w");

    fprintf(f, "core_idx,g1,g2\n");

    for (int row_num = 0; row_num < num_rows; row_num++)
    {
        for (int col_num = 0; col_num < num_cols; col_num++)
        {
            fprintf(f,"%f", measurements(row_num,col_num));

            if ( col_num < num_cols - 1 )
            {
                fprintf(f, ",");
            }
        }

        if ( row_num < num_rows - 1 )
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

inline double fitQuality(
    ArrX2d& points
)
{
    MatX2d centered = points.rowwise() - points.colwise().mean();
    MatX2d cov = (centered.adjoint() * centered) / double(points.rows() - 1);

    return 2 * sqrt(0.5 * cov.trace());
}

inline ArrX2d thresholdGuesses(ArrX2d xxs, double conf_frac)
{
    Eigen::VectorXd rrs = Eigen::sqrt(
        Eigen::pow(xxs.col(0).mean() - xxs.col(0), 2)
        +
        Eigen::pow(xxs.col(1).mean() - xxs.col(1), 2)
    );

    // std::cout << rrs << std::endl;

    int frac_boundary = conf_frac * xxs.rows();

    ArrX2d thresholded_points = ArrX2d(frac_boundary,2);
    Eigen::VectorXd thresholded_rrs = Eigen::VectorXd(frac_boundary);
    Eigen::VectorXi thresholded_idx = Eigen::VectorXi(frac_boundary);

    for (int idx = 0; idx < frac_boundary; idx++) { thresholded_idx[idx] = -1; }

    double min_range = 100;

    for (int range_idx = 0; range_idx < rrs.rows(); range_idx++)
    {
        if ( rrs[range_idx] < min_range )
        {
            min_range = rrs[range_idx];
            thresholded_points.row(0) = xxs.row(range_idx);
            thresholded_rrs[0] = rrs[range_idx];
            thresholded_idx[0] = range_idx;
        }
    }

    for (int row_idx = 1; row_idx < frac_boundary; row_idx++)
    {
        min_range = 100;

        for (int range_idx = 0; range_idx < rrs.rows(); range_idx++)
        {
            if ( rrs[range_idx] < min_range && rrs[range_idx] > thresholded_rrs[row_idx - 1] )
            {
                min_range = rrs[range_idx];
                thresholded_points.row(row_idx) = xxs.row(range_idx);
                thresholded_rrs[row_idx] = rrs[range_idx];
                thresholded_idx[row_idx] = range_idx;
            }
        }
    }

    // std::cout << thresholded_points << std::endl;

    return thresholded_points;
}

inline double polygonArea(ArrX2d vertexes)
{
    int og_rows = vertexes.rows();

    vertexes.conservativeResize(vertexes.rows() + 1, vertexes.cols());
    vertexes.row(vertexes.rows() - 1) = vertexes.row(0);

    double area = 0.;

    for (int row_idx = 0; row_idx < og_rows; row_idx++)
    {
        area += (vertexes(row_idx,0) * vertexes(row_idx + 1,1)) - (vertexes(row_idx + 1,0) * vertexes(row_idx,1));
    }

    area *= 0.5;

    return abs(area);
}

inline ArrX2d loopify(ArrX2d points)
{
    int og_rows = points.rows();

    points.conservativeResize(points.rows() + 1, points.cols());
    points.row(points.rows() - 1) = points.row(0);

    return points;
}

#endif /* __UTILS_H__ */
