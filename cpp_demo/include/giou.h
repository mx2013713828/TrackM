/*
 * File:        giou.h
 * Author:      Yufeng Ma
 * Date:        2024-06-01
 * Email:       97357473@qq.com
 * Description: This header file defines structures and functions for computing
 *              the generalized intersection over union (GIoU) and other related
 *              metrics for 3D bounding boxes. The main components include:
 *              - Bndbox: A structure representing a bounding box with position,
 *                dimensions, rotation, and additional properties.
 *              - Box3D: A structure representing a 3D bounding box, which can
 *                be initialized from Eigen vectors or Bndbox objects.
 *              - Functions for computing the 2D and 3D corners of bounding boxes,
 *                calculating the convex area, polygon area, and height of boxes,
 *                and performing the Sutherland-Hodgman clipping algorithm.
 *              - A function for calculating the intersection over union (IoU) 
 *                between two 3D bounding boxes.
 */

#ifndef GIOU_H
#define GIOU_H

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

struct Bndbox {
    float x;    // 中心点坐标
    float y;    // 中心点坐标
    float z;    // 中心点坐标
    float w;
    float l;
    float h;
    float rt;   // 航向 单位：弧度【-pi -- +pi】
    int id;
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};

struct Box3D {
    float x, y, z, w, l, h, yaw;
    int class_id;
    float score;
    int track_id;
    // 默认构造函数
    Box3D() : x(0), y(0), z(0), w(0), l(0), h(0), yaw(0), class_id(-1), score(0), track_id(-1) {}

    // 从 Eigen::VectorXd 初始化 Box3D
    Box3D(const Eigen::VectorXd& bbox, int class_id_ = -1, float score_ = 0, int track_id_ = -1) {
        x = bbox(0);
        y = bbox(1);
        z = bbox(2);
        w = bbox(3);
        l = bbox(4);
        h = bbox(5);
        yaw = bbox(6);
        class_id = class_id_;
        score = score_;
        track_id = track_id_;
    }

    // 从 Bndbox 初始化 Box3D beishan使用的检测结果结构
    Box3D(const Bndbox& bndbox) {
        x = bndbox.x;
        y = bndbox.y;
        z = bndbox.z;
        w = bndbox.w;
        l = bndbox.l;
        h = bndbox.h;
        yaw = bndbox.rt;
        class_id = bndbox.id;
        score = bndbox.score;
        track_id = -1;
    }

    // 拷贝构造函数
    Box3D(const Box3D& other) = default;
    Box3D(Box3D&& other) = default;
    Box3D& operator=(const Box3D& other) = default;
    Box3D& operator=(Box3D&& other) = default;
};

std::vector<std::array<float, 3>> box2corners(const Box3D& bbox);
float convex_area(const std::vector<std::array<float, 2>>& boxa_bottom, const std::vector<std::array<float, 2>>& boxb_bottom);
float compute_height(const std::vector<std::array<float, 3>>& corners1, const std::vector<std::array<float, 3>>& corners2, bool inter = true);
float polygon_area(const std::vector<std::array<float, 2>>& vertices);
std::vector<std::array<float, 2>> sutherland_hodgman_clip(const std::vector<std::array<float, 2>>& subject_polygon, const std::vector<std::array<float, 2>>& clip_polygon);
std::array<float, 3> calculate_iou(const Box3D& boxa_3d, const Box3D& boxb_3d);

#endif // GIOU_H
