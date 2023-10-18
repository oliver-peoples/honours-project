#ifndef __UTILS_H__
#define __UTILS_H__

#include <Eigen/Dense>

#include <cstdlib>
#include <iostream>
#include <stack>

typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector2d Vec2d;
typedef Eigen::MatrixX3d MatX3d;
typedef Eigen::MatrixX2d MatX2d;
typedef Eigen::Matrix3d Mat3d;
typedef Eigen::ArrayX3d ArrX3d;
typedef Eigen::ArrayX2d ArrX2d;
typedef Eigen::Array3d Arr3d;

const double CONF_FRAC = 1 - 1/sqrt(exp(1));
const double SQRT3 = sqrt(3.);

void createConcentricCores(ArrX3d& core_locations, int concentric_rings, double intercore_dist=1.)
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

void savePoints(std::string file_name, ArrX3d& points)
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

void savePoints(std::string file_name, ArrX2d& points)
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

void savePoints(std::string file_name, Eigen::Array<double,2,3>& points)
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

void saveIndexes(std::string file_name, Eigen::VectorXi indexes)
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

void saveMeasurements(std::string file_name, ArrX2d measurements)
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
void saveHeatmap(std::string file_name, Eigen::Array<double,rows,cols> measurements)
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
            else
            {
                fprintf(f, "\n");
            }
        }

        if ( row_num < num_rows - 1 )
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}

double fitQuality(
    ArrX2d& points
)
{
    MatX2d centered = points.rowwise() - points.colwise().mean();
    MatX2d cov = (centered.adjoint() * centered) / double(points.rows() - 1);

    return 2 * sqrt(0.5 * cov.trace());
}

ArrX2d multicoreMeasure(
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

struct MulticoreData
{
    ArrX3d& core_locations;
    ArrX2d& multicore_measure_noisy;
    Eigen::VectorXi& g2_capable_idx;
};


double multicoreChi2(
    const Eigen::VectorXd& vals_inp,
    Eigen::VectorXd* grad_out,
    void* opt_data
)
{
    MulticoreData* multicore_data_ptr = (MulticoreData*)opt_data;
    
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

    chi2.col(0) /= multicore_measure_noisy.col(0);
    chi2(g2_capable_idx,1) /= multicore_measure_noisy(g2_capable_idx,1);

    return chi2.col(0).sum() + chi2.col(1).sum();
}


struct Point

{

        double x;

        double y;

};

 

Point p0;

 

// A utility function to find next to top in a stack

Point nextToTop(std::stack<Point> &S)

{

    Point p = S.top();

    S.pop();

    Point res = S.top();

    S.push(p);

    return res;

}

 

// A utility function to swap two points

void swap(Point &p1, Point &p2)

{

    Point temp = p1;

    p1 = p2;

    p2 = temp;

}

 

// A utility function to return square of distance between p1 and p2

int dist(Point p1, Point p2)

{

    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);

}

 

int orientation(Point p, Point q, Point r)

{

    double val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

 

    if (val == 0)

        return 0; // colinear

    return (val > 0) ? 1 : 2; // clock or counterclock wise

}

 

// A function used by library function qsort() to sort an array of

// points with respect to the first point

int compare(const void *vp1, const void *vp2)

{

    Point *p1 = (Point *) vp1;

    Point *p2 = (Point *) vp2;

 

    // Find orientation

    int o = orientation(p0, *p1, *p2);

    if (o == 0)

        return (dist(p0, *p2) >= dist(p0, *p1)) ? -1 : 1;

 

    return (o == 2) ? -1 : 1;

}

 

// Prints convex hull of a set of n points.

ArrX2d convexHull(Point points[], int n)

{

    // Find the bottommost point

    double ymin = points[0].y;
    int min = 0;

    for (int i = 1; i < n; i++)

    {

        double y = points[i].y;

 

        // Pick the bottom-most or chose the left most point in case of tie

        if ((y < ymin) || (ymin == y && points[i].x < points[min].x))

            ymin = points[i].y, min = i;

    }

 

    // Place the bottom-most point at first position

    swap(points[0], points[min]);

 

    // Sort n-1 points with respect to the first point.  A point p1 comes

    // before p2 in sorted ouput if p2 has larger polar angle (in

    // counterclockwise direction) than p1

    p0 = points[0];

    std::qsort(&points[1], n - 1, sizeof(Point), compare);

 

    // Create an empty stack and push first three points to it.

    std::stack<Point> S;

    S.push(points[0]);

    S.push(points[1]);

    S.push(points[2]);

 

    // Process remaining n-3 points

    for (int i = 3; i < n; i++)

    {

        // Keep removing top while the angle formed by points next-to-top,

        // top, and points[i] makes a non-left turn

        while (orientation(nextToTop(S), S.top(), points[i]) != 2)

            S.pop();

        S.push(points[i]);

    }

 

    // Now stack has the output points, print contents of stack

    int idx = 0;

    ArrX2d hull_points = ArrX2d(S.size(),2);

    // std::cout << S.size() << std::endl;

    while (!S.empty())

    {

        Point p = S.top();

        hull_points(idx,0) = p.x;
        hull_points(idx,1) = p.y;

        // std::cout << "(" << p.x << ", " << p.y << ")" << std::endl;

        S.pop();

        idx++;
    }

    return hull_points;
}

ArrX2d convexHull(ArrX2d& optim_points)
{
    Point* points = (Point*)malloc(sizeof(Point) * optim_points.rows());

    int n = optim_points.rows();

    for (int point_idx = 0; point_idx < n; point_idx++)
    {
        points[point_idx] = { optim_points(point_idx,0),optim_points(point_idx,1) };
    }

    return convexHull(points, n);
}

ArrX2d thresholdGuesses(ArrX2d xxs, double conf_frac)
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

double polygonArea(ArrX2d vertexes)
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

 
// emitter_distances = zeros(length(cores),2);

// for emitter_idx=1:2
//     diffv = cores - emitter_xy(emitter_idx,:);
    
//     emitter_distances(:,emitter_idx) = sqrt(diffv(:,1).^2+diffv(:,2).^2+diffv(:,3).^2);
// end

// powers = exp(-(emitter_distances.^2/2)/(2*core_psf^2));

// powers(:,1) = powers(:,1) * emitter_brightness(1);
// powers(:,2) = powers(:,2) * emitter_brightness(2);

// g1_pred = (powers(:,1)+powers(:,2))./(emitter_brightness(1)+emitter_brightness(2));

// alpha = powers(g2_capable_idx,1)./powers(g2_capable_idx,2);

// g2_pred = (2 * alpha)./((1+alpha).^2);

#endif /* __UTILS_H__ */
