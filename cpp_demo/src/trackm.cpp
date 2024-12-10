
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
    : Filter(bbox3D, info, Track_ID), kf(11, 7) {
    _init_kalman_filter();
}

void KF::_init_kalman_filter() {
    // kf.F(0, 7) = kf.F(1, 8) = kf.F(2, 9) = 1;
    double dt = 0.1;

    kf.F(0, 7) = dt; 
    kf.F(1, 8) = dt; 
    kf.F(2, 9) = dt; 
    kf.F(6, 10) = dt; // yaw 更新 (yaw = yaw + yaw_rate * dt)

    for (int i = 0; i < 7; ++i) {
        kf.H(i, i) = 1;
    }

    kf.P.bottomRightCorner(4, 4) *= 100; // 加大速度相关的不确定性
    kf.P *= 10;

    // 过程噪声协方差矩阵 Q
    kf.Q.bottomRightCorner(4, 4) *= 1; // 增大速度过程噪声，提高系统对变化的敏感性
    // kf.Q(10, 10) = 0.1;
    
    // 测量噪声协方差矩阵 R
    kf.R *= 0.1; // 减小测量噪声，增强对传感器数据的信任

    kf.x.head<7>() = initial_pos;

    Box3D initial_box(initial_pos, this->info["class_id"],this->info["score"],this->track_id);

    track_history.push_back(initial_box);
}

void KF::predict() {
    kf.predict();
}
Eigen::VectorXd KF::get_state() const {
    return kf.x;
}

Eigen::VectorXd KF::get_velocity() const {
    return kf.x.tail<3>();
}

float KF::get_yaw_speed() const {
    // 获取状态向量中的速度分量 vx 和 vy
    float vx = kf.x(7); // 
    float vy = kf.x(8); // 
    float yaw = kf.x(6); // 

    // 计算沿 yaw 方向的速度    
    float yaw_speed = vx * std::cos(yaw) + vy * std::sin(yaw);

    // m/s
    return yaw_speed;
}

const std::vector<Box3D>& KF::get_history() const {
    return track_history;
}

const std::vector<Box3D>& KF::track_prediction(int steps) {
    track_future.clear();  // 清空之前的预测记录
    EKalmanFilter future_kf = kf;

    for (int i = 0; i < steps; ++i) {
        future_kf.predict();
        track_future.push_back(future_kf.x);  // 假设 Box3D 构造函数接受状态向量
    }

    return track_future;  // 返回 const 引用
}

// const std::array<std::pair<float, float>, 20>& KF::track_prediction(int steps) {
//     // 限制步数，不能超过数组的容量
//     int future_steps = std::min(steps, static_cast<int>(track_future.size()));

//     // 创建一个临时的 EKF 作为未来预测的状态
//     EKalmanFilter future_kf = kf;

//     for (int i = 0; i < future_steps; ++i) {
//         future_kf.predict(); // 执行一次预测
//         track_future[i] = {future_kf.x(0), future_kf.x(1)}; // 只存储 x, y 坐标
//     }

//     // // 填充多余的点为默认值，保持数组的一致性
//     // for (int i = future_steps; i < track_future.size(); ++i) {
//     //     track_future[i] = {0.0, 0.0};
//     // }

//     return track_future;
// }


void KF::update(const Eigen::VectorXd& bbox3D, float confidence) {
    // 获取卡尔曼滤波器中的 yaw 值
    double previous_yaw = kf.x(6);
    double new_yaw = bbox3D(6);

    // 计算 yaw 的差值并限制在 [-pi, pi] 范围内
    double yaw_diff = limit_angle(new_yaw - previous_yaw);
    bool large_yaw_change = std::abs(yaw_diff) > M_PI / 2;

    if (large_yaw_change) {
        // 判断 hits 是否较少（例如，hits 小于 3 时认为刚开始）
        if (this->hits < 6) {
            // 根据置信度高低决定采用哪个 yaw
            if (confidence > this->prev_confidence) {
                std::cout << "Adopting new yaw (" << new_yaw << ") due to higher confidence (" 
                          << confidence << " > " << this->prev_confidence << ")" << std::endl;
                // 采用新yaw
            } else if (confidence < this->prev_confidence) {
                std::cout << "Keeping previous yaw (" << previous_yaw << ") due to higher previous confidence (" 
                          << this->prev_confidence << " > " << confidence << ")" << std::endl;
                new_yaw = previous_yaw;  // 保持原来的yaw
            } else {
                std::cout << "Similar confidence levels, smoothing yaw transition between " 
                          << previous_yaw << " and " << new_yaw << std::endl;
                new_yaw = (previous_yaw + new_yaw) / 2;  // 平滑过渡
            }
        } else {
            // 当 hits 较多时，忽略大的 yaw 变化，保持之前的状态
            std::cout << "Large yaw change detected: " << previous_yaw << " -> " << new_yaw 
                      << ", ignoring update due to stability." << std::endl;
            new_yaw = previous_yaw;  // 忽略更新
        }
    }

    // 使用平滑后的 yaw 值更新 bbox3D 的 yaw
    Eigen::VectorXd smoothed_bbox = bbox3D;
    smoothed_bbox(6) = previous_yaw + limit_angle(new_yaw - previous_yaw);

    // 更新卡尔曼滤波器状态
    kf.update(smoothed_bbox);

    // 确保更新后的 yaw 值在 [-pi, pi] 范围内
    kf.x(6) = limit_angle(kf.x(6));

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

void print_results(const std::vector<std::array<int, 2>>& matches,
                   const std::vector<int>& unmatched_detections,
                   const std::vector<int>& unmatched_trackers) {
    std::cout << "Matches:" << std::endl;
    for (const auto& match : matches) {
        std::cout << "Detection " << match[0] << " matched with Tracker " << match[1] << std::endl;
    }

    std::cout << "Unmatched Detections:" << std::endl;
    for (int idx : unmatched_detections) {
        std::cout << "Detection " << idx << std::endl;
    }

    std::cout << "Unmatched Trackers:" << std::endl;
    for (int idx : unmatched_trackers) {
        std::cout << "Tracker " << idx << std::endl;
    }
}
