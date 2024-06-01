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
    int id;
    float score;

    // 默认构造函数
    Box3D() : x(0), y(0), z(0), w(0), l(0), h(0), yaw(0), id(-1), score(0) {}

    // 从 Eigen::VectorXd 初始化 Box3D
    Box3D(const Eigen::VectorXd& bbox, int id_ = -1, float score_ = 0) {
        x = bbox(0);
        y = bbox(1);
        z = bbox(2);
        w = bbox(3);
        l = bbox(4);
        h = bbox(5);
        yaw = bbox(6);
        id = id_;
        score = score_;
    }

    // 从 Bndbox 初始化 Box3D
    Box3D(const Bndbox& bndbox) {
        x = bndbox.x;
        y = bndbox.y;
        z = bndbox.z;
        w = bndbox.w;
        l = bndbox.l;
        h = bndbox.h;
        yaw = bndbox.rt;
        id = bndbox.id;
        score = bndbox.score;
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
