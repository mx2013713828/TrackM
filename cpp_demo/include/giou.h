/*
 * File:        giou.h
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: 3D bounding box IoU and GIoU calculation.
 */

#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "common_types.h"

std::vector<std::array<float, 3>> box2corners(const Box3D& bbox);
float convex_area(const std::vector<std::array<float, 2>>& boxa_bottom, const std::vector<std::array<float, 2>>& boxb_bottom);
float compute_height(const std::vector<std::array<float, 3>>& corners1, const std::vector<std::array<float, 3>>& corners2, bool inter = true);
float polygon_area(const std::vector<std::array<float, 2>>& vertices);
std::vector<std::array<float, 2>> sutherland_hodgman_clip(const std::vector<std::array<float, 2>>& subject_polygon, const std::vector<std::array<float, 2>>& clip_polygon);
std::array<float, 3> calculate_iou(const Box3D& boxa_3d, const Box3D& boxb_3d);

// 基于yaw角度差异的增强GIOU计算
std::array<float, 3> calculate_iou_with_yaw(const Box3D& boxa_3d, const Box3D& boxb_3d, float yaw_weight = 0.5);
