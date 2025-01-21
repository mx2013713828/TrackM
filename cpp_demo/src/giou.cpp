/*
 * File:        giou.cpp
 * Author:      Yufeng Ma
 * Date:        2025-01-20
 * Email:       97357473@qq.com
 * Description: Implementation of 3D IoU and GIoU calculations.
 */

#include "../include/giou.h"
#include <algorithm>
#include <numeric>
#include <limits>
#include <opencv2/opencv.hpp>

#include "../include/giou.h"

std::vector<std::array<float, 3>> box2corners(const Box3D& bbox) {
    float yaw = bbox.yaw;
    float c = std::cos(yaw);
    float s = std::sin(yaw);

    // std::cout<< "cos:"<<c <<"sin:"<<s<<std::endl;

    std::array<std::array<float, 3>, 3> R = {{
        {c, -s, 0},
        {s, c, 0},
        {0, 0, 1}
    }};
    
    float x = bbox.x, y = bbox.y, z = bbox.z;
    float l = bbox.l, w = bbox.w, h = bbox.h;
    // std::cout<<"x:"<<x<<"y:"<<y<<"z:"<<z<<"w:"<<w<<"l:"<<l<<"h:"<<h<<std::endl;

    std::vector<float> y_corners = {l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2};
    std::vector<float> x_corners = {w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2};
    std::vector<float> z_corners = {h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2};

    std::vector<std::array<float, 3>> corners_3d(8);
    for (int i = 0; i < 8; ++i) {
        std::array<float, 3> point = {x_corners[i], y_corners[i], z_corners[i]};
        for (int j = 0; j < 3; ++j) {
            corners_3d[i][j] = R[j][0] * point[0] + R[j][1] * point[1] + R[j][2] * point[2];
        }
        corners_3d[i][0] += x;
        corners_3d[i][1] += y;
        corners_3d[i][2] += z;
    }

    return corners_3d;
}

float convex_area(const std::vector<std::array<float, 2>>& boxa_bottom, const std::vector<std::array<float, 2>>& boxb_bottom) {
    // float xc1 = std::min(
    //         std::min_element(boxa_bottom.begin(), boxa_bottom.end(), [](auto& a, auto& b) { return a[0] < b[0]; })->at(0),
    //         std::min_element(boxb_bottom.begin(), boxb_bottom.end(), [](auto& a, auto& b) { return a[0] < b[0]; })->at(0)
    //     );
    // float yc1 = std::min(
    //     std::min_element(boxa_bottom.begin(), boxa_bottom.end(), [](auto& a, auto& b) { return a[1] < b[1]; })->at(1),
    //     std::min_element(boxb_bottom.begin(), boxb_bottom.end(), [](auto& a, auto& b) { return a[1] < b[1]; })->at(1)
    // );
    // float xc2 = std::max(
    //     std::max_element(boxa_bottom.begin(), boxa_bottom.end(), [](auto& a, auto& b) { return a[0] < b[0]; })->at(0),
    //     std::max_element(boxb_bottom.begin(), boxb_bottom.end(), [](auto& a, auto& b) { return a[0] < b[0]; })->at(0)
    // );
    // float yc2 = std::max(
    //     std::max_element(boxa_bottom.begin(), boxa_bottom.end(), [](auto& a, auto& b) { return a[1] < b[1]; })->at(1),
    //     std::max_element(boxb_bottom.begin(), boxb_bottom.end(), [](auto& a, auto& b) { return a[1] < b[1]; })->at(1)
    // );
    
    // return (xc2 - xc1) * (yc2 - yc1);

    // test log
    // std::cout<<"boxa:"<<std::endl;
    // for(auto& a : boxa_bottom) {
    //     std::cout<< a[0] <<","<< a[1] <<std::endl;
    // }

    // std::cout<<"boxb:"<<std::endl;
    // for(auto& b : boxb_bottom) {
    //     std::cout<< b[0] <<","<< b[1] <<std::endl;
    // }
    // 第二种使用凸包算法的实现
    std::vector<std::array<float, 2>> all_corners = boxa_bottom;
    all_corners.insert(all_corners.end(), boxb_bottom.begin(), boxb_bottom.end());

    // 使用OpenCV的凸包算法
    std::vector<cv::Point2f> points;
    for (const auto& corner : all_corners) {
        points.emplace_back(corner[0], corner[1]);
    }
    std::vector<int> hull_indices;
    cv::convexHull(points, hull_indices);

    std::vector<std::array<float, 2>> hull_points;
    for (int idx : hull_indices) {
        hull_points.push_back({points[idx].x, points[idx].y});
    }

    float hull_area = polygon_area(hull_points);

    // 返回凸包面积
    return hull_area;

}

float compute_height(const std::vector<std::array<float, 3>>& corners1, const std::vector<std::array<float, 3>>& corners2, bool inter) {
    if (inter) {
        float zmax = std::min(corners1[0][2], corners2[0][2]);
        float zmin = std::max(corners1[4][2], corners2[4][2]);
        return std::max(0.0f, zmax - zmin);
    } else {
        float zmax = std::max(corners1[0][2], corners2[0][2]);
        float zmin = std::min(corners1[4][2], corners2[4][2]);
        return std::max(0.0f, zmax - zmin);
    }
}

float polygon_area(const std::vector<std::array<float, 2>>& vertices) {
    int n = vertices.size();
    float area = 0.0f;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += vertices[i][0] * vertices[j][1];
        area -= vertices[i][1] * vertices[j][0];
    }
    return std::abs(area) / 2.0f;
}

std::vector<std::array<float, 2>> sutherland_hodgman_clip(const std::vector<std::array<float, 2>>& subject_polygon, const std::vector<std::array<float, 2>>& clip_polygon) {
    auto inside = [](const std::array<float, 2>& p, const std::array<float, 2>& edge_start, const std::array<float, 2>& edge_end) {
        return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) >= (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0]);
    };

    auto compute_intersection = [](const std::array<float, 2>& p1, const std::array<float, 2>& p2, const std::array<float, 2>& p3, const std::array<float, 2>& p4) {
        float denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]);
        if (denom == 0.0f) return std::array<float, 2>{std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};
        float x = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denom;
        float y = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denom;
        return std::array<float, 2>{x, y};
    };

    std::vector<std::array<float, 2>> output_list = subject_polygon;
    for (int i = 0; i < clip_polygon.size(); ++i) {
        std::vector<std::array<float, 2>> input_list = output_list;
        output_list.clear();
        if (input_list.empty()) break;
        std::array<float, 2> edge_start = clip_polygon[i];
        std::array<float, 2> edge_end = clip_polygon[(i + 1) % clip_polygon.size()];
        for (int j = 0; j < input_list.size(); ++j) {
            std::array<float, 2> current_point = input_list[j];
            std::array<float, 2> prev_point = input_list[(j - 1 + input_list.size()) % input_list.size()];
            if (inside(current_point, edge_start, edge_end)) {
                if (!inside(prev_point, edge_start, edge_end)) {
                    output_list.push_back(compute_intersection(prev_point, current_point, edge_start, edge_end));
                }
                output_list.push_back(current_point);
            } else if (inside(prev_point, edge_start, edge_end)) {
                output_list.push_back(compute_intersection(prev_point, current_point, edge_start, edge_end));
            }
        }
    }
    return output_list;
}

std::array<float, 3> calculate_iou(const Box3D& boxa_3d, const Box3D& boxb_3d) {
    auto corners_a = box2corners(boxa_3d);
    auto corners_b = box2corners(boxb_3d);

    std::vector<std::array<float, 2>> boxa_bot = {
        {corners_a[7][0], corners_a[7][1]},
        {corners_a[6][0], corners_a[6][1]},
        {corners_a[5][0], corners_a[5][1]},
        {corners_a[4][0], corners_a[4][1]}};

    std::vector<std::array<float, 2>> boxb_bot = {
        {corners_b[7][0], corners_b[7][1]},
        {corners_b[6][0], corners_b[6][1]},
        {corners_b[5][0], corners_b[5][1]},
        {corners_b[4][0], corners_b[4][1]}};

    auto intersection_2d = sutherland_hodgman_clip(boxa_bot, boxb_bot);
    float I_2D = intersection_2d.empty() ? 0.0f : polygon_area(intersection_2d);
    float C_2D = convex_area(boxa_bot, boxb_bot);

    float h_overlap = compute_height(corners_a, corners_b, true);
    float h_union = compute_height(corners_a, corners_b, false);

    float I_3D = I_2D * h_overlap;
    float C_3D = C_2D * h_union;

    float U_2D = boxa_3d.l * boxa_3d.w + boxb_3d.l * boxb_3d.w - I_2D;
    float U_3D = boxa_3d.l * boxa_3d.w * boxa_3d.h + boxb_3d.l * boxb_3d.w * boxb_3d.h - I_3D;
    
    // std::cout << "Box A Bottom Corners:" << std::endl;
    // for (const auto& point : boxa_bot) {
    //     std::cout << "(" << point[0] << ", " << point[1] << ")" << std::endl;
    // }

    // std::cout << "Box B Bottom Corners:" << std::endl;
    // for (const auto& point : boxb_bot) {
    //     std::cout << "(" << point[0] << ", " << point[1] << ")" << std::endl;
    // }


    // std::cout << "I_2D: " << I_2D << std::endl;
    // std::cout << "I_3D: " << I_3D << std::endl;
    // std::cout << "C_3D: " << C_3D << std::endl;
    // std::cout << "U_2D: " << U_2D << std::endl;
    // std::cout << "U_3D: " << U_3D << std::endl;
    
    float IOU2D = I_2D / U_2D;
    float IOU3D = I_3D / U_3D;
    float GIOU = IOU3D - (C_3D - U_3D) / C_3D;

    return {GIOU, IOU3D, IOU2D};
}
