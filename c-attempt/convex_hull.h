#ifndef __CONVEX_HULL_H__
#define __CONVEX_HULL_H__

#include "utils.h"

// https://algoteka.com/samples/35/graham-scan-convex-hull-algorithm-c-plus-plus-o%2528n-log-n%2529-readable-solution

#include <cstdio>
#include <stack>
#include <algorithm>
using namespace std;

class Point    {
public:
    double x, y;

    // comparison is done first on y coordinate and then on x coordinate
    bool operator < (Point b) {
        if (y != b.y)
            return y < b.y;
        return x < b.x;
    }
};

struct Vector2D {
    double x;
    double y;
    
    Vector2D operator-(Vector2D r) {
        return {x - r.x, y - r.y};
    }
    double operator*(Vector2D r) {
        return x * r.x + y * r.y;
    }
    Vector2D rotate90() {  // Rotate 90 degrees counter-clockwise
        return {-y, x};
    }
    double manhattan_length() {
        return abs(x) + abs(y);
    }
    bool operator==(Vector2D r) {
        return x == r.x && y == r.y;
    }
    bool operator!=(Vector2D r) {
        return x != r.x || y != r.y;
    }
};


inline std::vector<Vector2D> graham_scan(std::vector<Vector2D> points) {
    Vector2D first_point = *std::min_element(points.begin(), points.end(), [](Vector2D &left, Vector2D &right) {
        return std::make_tuple(left.y, left.x) < std::make_tuple(right.y, right.x);
    });  // Find the lowest and leftmost point
    
    std::sort(points.begin(), points.end(), [&](Vector2D &left, Vector2D &right) {
        if(left == first_point) {
            return right != first_point;
        } else if (right == first_point) {
            return false;
        }
        double dir = (left-first_point).rotate90() * (right-first_point);
        if(dir == 0) {  // If the points are on a line with first point, sort by distance (manhattan is equivalent here)
            return (left-first_point).manhattan_length() < (right-first_point).manhattan_length();
        }
        return dir > 0;
        // Alternative approach, closer to common algorithm formulation but inferior:
        // return atan2(left.y - first_point.y, left.x - first_point.x) < atan2(right.y - first_point.y, right.x - first_point.x);
    });  // Sort the points by angle to the chosen first point
    
    std::vector<Vector2D> result;
    for(auto pt : points) {
        // For as long as the last 3 points cause the hull to be non-convex, discard the middle one
        while (result.size() >= 2 &&
               (result[result.size()-1] - result[result.size()-2]).rotate90() * (pt - result[result.size()-1]) <= 0) {
            result.pop_back();
        }
        result.push_back(pt);
    }
    return result;
}

inline ArrX2d convexHull(ArrX2d& optim_points)
{
    int n = optim_points.rows();

    std::vector<Vector2D> pts;

    for (int point_idx = 0; point_idx < n; point_idx++)
    {
        pts.push_back({ optim_points(point_idx,0),optim_points(point_idx,1) });
    }

    std::vector<Vector2D> hull_vertex_list = graham_scan(pts);

    int n_vertices = hull_vertex_list.size();

    ArrX2d convex_hull = ArrX2d(n_vertices,2);

    for (int vertex_idx = 0; vertex_idx < n_vertices; vertex_idx++)
    {
        convex_hull(vertex_idx,0) = hull_vertex_list[vertex_idx].x;
        convex_hull(vertex_idx,1) = hull_vertex_list[vertex_idx].y;
    }

    return convex_hull;
}

#endif /* __CONVEX_HULL_H__ */
