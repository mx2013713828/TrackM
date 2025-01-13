/*
 * File:        trackm.cpp
 * Author:      Yufeng Ma
 * Date:        2024-06-01
 * Email:       97357473@qq.com
 * Description: Implementation of the tracking filter classes and matching algorithm.
 */

#include "../include/trackm.h"
#include "../include/kalman_filter.h"
#include <algorithm>
#include <iostream>
#include <tuple>
#include <Eigen/Dense>
#include "../include/giou.h"
#include <numeric>
#include <vector>
#include <unordered_map>


// 将角度限制在 [-pi, pi] 范围内
double limit_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

Filter::Filter(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int Track_ID)
    : initial_pos(bbox3D), time_since_update(0), track_id(Track_ID), hits(1), info(info) {}

KF::KF(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int Track_ID)
    : Filter(bbox3D, info, Track_ID), ekf(19, 11) {
    _init_kalman_filter();
}

void KF::_init_kalman_filter() {
    double dt = 0.1;

    // 1. 状态转移矩阵 F 初始化
    // 车辆坐标系部分
    ekf.F(0, 7) = dt;   // x_world 对 vx_world 的影响
    ekf.F(1, 8) = dt;   // y_world 对 vy_world 的影响
    ekf.F(2, 9) = dt;   // z_world 对 vz_world 的影响
    ekf.F(6, 10) = dt;  // heading_world 对 v_heading_world 的影响

    // 大地坐标系部分
    ekf.F(11, 15) = dt;  // x_earth 对 vx_earth 的影响
    ekf.F(12, 16) = dt;  // y_earth 对 vy_earth 的影响
    ekf.F(13, 17) = dt;  // z_earth 对 vz_earth 的影响
    ekf.F(14, 18) = dt;  // heading_earth 对 v_heading_earth 的影响

    // 2. 测量矩阵 H 初始化 (14x19)
    // 车辆坐标系观测 (7维)
    for (int i = 0; i < 7; ++i) {
        ekf.H(i, i) = 1;  // x,y,z,w,l,h,heading_world
    }
    // 大地坐标系观测 (4维)
    ekf.H(7, 11) = 1;   // x_earth
    ekf.H(8, 12) = 1;   // y_earth
    ekf.H(9, 13) = 1;   // z_earth
    ekf.H(10, 14) = 1;  // heading_earth

    // 3. 状态协方差矩阵 P 初始化 - 调整初始不确定性
    ekf.P = Eigen::MatrixXd::Identity(19, 19);
    ekf.P.topLeftCorner(7, 7) *= 1.0;  // 降低位置和尺寸的不确定性
    ekf.P.block<4, 4>(7, 7) *= 10.0;   // 速度的不确定性适中
    ekf.P.block<4, 4>(11, 11) *= 1.0;  // 降低大地坐标系位置的不确定性
    ekf.P.block<4, 4>(15, 15) *= 10.0; // 速度的不确定性适中

    // 4. 过程噪声协方差矩阵 Q 初始化 - 调小过程噪声
    ekf.Q = Eigen::MatrixXd::Identity(19, 19) * 0.1;
    ekf.Q.block<4, 4>(7, 7) *= 1.0;   // 速度过程噪声适中
    ekf.Q(10, 10) = 0.1;              // heading_world 速度噪声较小
    ekf.Q.block<4, 4>(15, 15) *= 1.0; // 大地坐标系速度噪声适中
    ekf.Q(18, 18) = 0.1;              // heading_earth 速度噪声较小

    // 5. 测量噪声协方差矩阵 R 初始化 - 调整测量噪声
    ekf.R = Eigen::MatrixXd::Identity(11, 11) * 0.1;  // 降低整体测量噪声

    // 6. 状态向量初始化
    ekf.x = Eigen::VectorXd::Zero(19);
    
    // 车辆坐标系状态初始化
    ekf.x.head<7>() = initial_pos.head<7>();  // 前7维是车辆坐标系状态
    
    // 车辆坐标系速度初始化为0
    ekf.x.segment<4>(7).setZero();  // vx,vy,vz,v_heading_world
    
    // 大地坐标系位置初始化
    ekf.x.segment<4>(11) = initial_pos.tail<4>();  // 后4维是大地坐标系状态
    
    // 大地坐标系速度初始化为0
    ekf.x.segment<4>(15).setZero();  // vx,vy,vz,v_heading_earth

    std::cout << "Initial state: " << ekf.x.transpose() << std::endl;
}

void KF::predict() {
    ekf.predict();
}
Eigen::VectorXd KF::get_state() const {
    return ekf.x;
}

Eigen::VectorXd KF::get_velocity() const {
    return ekf.x.tail<3>();
}

float KF::get_yaw_speed() const {
    // 获取状态向量中的速度分量 vx 和 vy
    float vx = ekf.x(7); // 
    float vy = ekf.x(8); // 
    float yaw = ekf.x(6); // 

    // 计算沿 yaw 方向的速度    
    float yaw_speed = vx * std::cos(yaw) + vy * std::sin(yaw);

    // m/s
    return yaw_speed;
}

const std::vector<Box3D>& KF::get_history() const {
    return track_history;
}

std::vector<point_t> KF::track_world_prediction(int steps) const {
    std::vector<point_t> predictions;
    EKalmanFilter future_kf = ekf;  // 创建临时副本进行预测

    for (int i = 0; i < steps; ++i) {
        future_kf.predict();
        
        point_t pred_point;
        pred_point.x = future_kf.x(0);  // x_world
        pred_point.y = future_kf.x(1);  // y_world
        predictions.push_back(pred_point);
    }

    return predictions;
}

std::vector<point_t> KF::track_earth_prediction(int steps) const {
    std::vector<point_t> predictions;
    EKalmanFilter future_kf = ekf;  // 创建临时副本进行预测

    for (int i = 0; i < steps; ++i) {
        future_kf.predict();
        
        point_t pred_point;
        pred_point.x = future_kf.x(11);  // x_earth
        pred_point.y = future_kf.x(12);  // y_earth
        predictions.push_back(pred_point);
    }

    return predictions;
}

Eigen::VectorXd KF::get_world_state() const {
    return ekf.x.head<7>();  // 返回车辆坐标系状态
}

Eigen::VectorXd KF::get_earth_state() const {
    Eigen::VectorXd earth_state(4);  // x, y, z, heading
    earth_state << ekf.x.segment<3>(11), ekf.x(14);  // 返回大地坐标系状态
    return earth_state;
}

void KF::update(const target_t& detection, float confidence) {
    // 更新不需要卡尔曼滤波的属性
    info["x_pixel"] = detection.x_pixel;
    info["y_pixel"] = detection.y_pixel;
    info["w_pixel"] = detection.w_pixel;
    info["h_pixel"] = detection.h_pixel;
    info["time_stamp"] = detection.time_stamp;
    info["property"] = detection.property;
    info["k"] = detection.k;
    info["s"] = detection.s;
    // 更新分类和置信度
    info["class_id"] = detection.classid;
    info["score"] = detection.conf;

    points_world = detection.points_world;
    points_earth = detection.points_earth;

    info["x_world1"] = detection.x_world1;
    info["y_world1"] = detection.y_world1;
    info["w_world1"] = detection.w_world1;
    info["h_world1"] = detection.h_world1;
    info["l_world1"] = detection.l_world1;

    info["x_world2"] = detection.x_world2;
    info["y_world2"] = detection.y_world2;
    info["w_world2"] = detection.w_world2;
    info["h_world2"] = detection.h_world2;
    info["l_world2"] = detection.l_world2;

    // 构建观测向量
    Eigen::VectorXd z(11);  // 修改为 11 维
    
    // 车辆坐标系观测 (7维)
    z(0) = detection.x_world;
    z(1) = detection.y_world;
    z(2) = detection.z_world;
    z(3) = detection.w_world;
    z(4) = detection.l_world;
    z(5) = detection.h_world;
    z(6) = detection.heading_world;

    // 大地坐标系观测 (4维)
    z(7) = detection.x_earth;
    z(8) = detection.y_earth;
    z(9) = detection.z_earth;
    z(10) = detection.heading_earth;

    // 获取卡尔曼滤波器中的 yaw 值
    double previous_yaw = ekf.x(6);
    double new_yaw = detection.heading_world;

    // 计算 yaw 的差值并限制在 [-pi, pi] 范围内
    double yaw_diff = limit_angle(new_yaw - previous_yaw);
    bool large_yaw_change = std::abs(yaw_diff) > M_PI / 6;

    if (large_yaw_change) {
        if (this->hits < 6) {
            if (confidence > this->prev_confidence) {
                // 采用新yaw
            } else if (confidence < this->prev_confidence) {
                new_yaw = previous_yaw;
            } else {
                std::cout<<"large_yaw_change:平滑过渡"<<std::endl;
                new_yaw = (previous_yaw + new_yaw) / 2;  // 平滑过渡
            }
        } else {
            // 对于稳定的跟踪，使用更保守的更新
            new_yaw = previous_yaw;
        }
    }
    z(6) = previous_yaw + limit_angle(new_yaw - previous_yaw);
    z(10) = ekf.x(14) + limit_angle(detection.heading_earth - ekf.x(14));
    // 更新卡尔曼滤波器状态
    ekf.update(z);
    // 确保更新后的航向角在 [-pi, pi] 范围内
    ekf.x(6) = limit_angle(ekf.x(6));
    ekf.x(14) = limit_angle(ekf.x(14));

    // std::cout << "After update, state: " << ekf.x.transpose() << std::endl;  // 调试信息

    // 更新 prev_confidence 为当前帧的置信度
    this->prev_confidence = confidence;
}



std::tuple<std::vector<std::array<int, 2>>, std::vector<int>, std::vector<int>>
associate_detections_to_trackers(const std::vector<Box3D>& detections,
                                 const std::vector<Box3D>& trackers,
                                 float iou_threshold) {
    if (trackers.empty()) {
        return std::make_tuple(std::vector<std::array<int, 2>>(), 
                               std::vector<int>(detections.size()), 
                               std::vector<int>());
    }

    Eigen::MatrixXf iou_matrix = Eigen::MatrixXf::Zero(detections.size(), trackers.size());

    for (size_t d = 0; d < detections.size(); ++d) {
        for (size_t t = 0; t < trackers.size(); ++t) {
            Box3D boxa_3d = detections[d];
            Box3D boxb_3d = trackers[t];
            auto [giou, iou3d, iou2d] = calculate_iou(boxa_3d, boxb_3d);
            // std::cout<<"giou: "<<giou<<std::endl;
            iou_matrix(d, t) = giou;
        }
    }

    std::vector<int> row_indices(detections.size());
    std::vector<int> col_indices(trackers.size());
    std::iota(row_indices.begin(), row_indices.end(), 0);
    std::iota(col_indices.begin(), col_indices.end(), 0);

    std::sort(row_indices.begin(), row_indices.end(), [&iou_matrix](int i1, int i2) {
        return iou_matrix.row(i1).maxCoeff() > iou_matrix.row(i2).maxCoeff();
    });

    std::vector<std::array<int, 2>> matches;
    std::vector<int> unmatched_detections, unmatched_trackers;

    for (int i : row_indices) {
        int best_j = -1;
        float best_iou = -1.0;

        for (int j : col_indices) {
            if (iou_matrix(i, j) > best_iou) {
                best_iou = iou_matrix(i, j);
                best_j = j;
            }
        }

        if (best_iou >= iou_threshold) {
            matches.push_back({i, best_j});
            col_indices.erase(std::remove(col_indices.begin(), col_indices.end(), best_j), col_indices.end());
        } else {
            unmatched_detections.push_back(i);
        }
    }

    for (int j : col_indices) {
        unmatched_trackers.push_back(j);
    }

    return std::make_tuple(matches, unmatched_detections, unmatched_trackers);
}


