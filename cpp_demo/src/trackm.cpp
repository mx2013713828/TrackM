/*
 * File:        trackm.cpp
 * Author:      Yufeng Ma
 * Date:        2025-01-20
 * Email:       97357473@qq.com
 * Description: Core tracking filter implementation and state estimation.
 */

#include "../include/trackm.h"
#include "../include/kalman_filter.h"
#include "../include/ekf.h"
#include "../include/iekf.h"
#include <algorithm>
#include <iostream>
#include <numeric>  // 为 std::iota

// 将角度限制在 [-pi, pi] 范围内
double limit_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

Filter::Filter(const Eigen::VectorXd& bbox3D, 
              const std::unordered_map<std::string, float>& info, 
              int Track_ID,
              FilterType filter_type)
    : initial_pos(bbox3D), time_since_update(0), track_id(Track_ID), 
      hits(1), info(info), prev_confidence(0.0) {
    // 添加调试信息
    // std::cout << "Creating new filter with ID: " << Track_ID << std::endl;
    // std::cout << "Initial position: " << bbox3D.transpose() << std::endl;
    
    init_filter(filter_type);
}

void Filter::init_filter(FilterType filter_type) {
    // std::cout << "Initializing filter with type: " 
    //           << (filter_type == FilterType::EKF ? "EKF" : 
    //               filter_type == FilterType::KF ? "KF" : "IEKF") 
    //           << std::endl;
              
    switch (filter_type) {
        case FilterType::KF:
            filter = std::make_unique<KalmanFilter>(19, 11);
            break;
        case FilterType::EKF:
            filter = std::make_unique<ExtendedKalmanFilter>(19, 11);
            break;
        case FilterType::IEKF:
            filter = std::make_unique<IteratedExtendedKalmanFilter>(19, 11);
            break;
    }
    _init_kalman_filter();
}
    
void Filter::_init_kalman_filter() {
    double dt = 0.1;
    
    // 1. 设置状态转移矩阵 F
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(19, 19);
    
    // 车辆坐标系部分
    F(0, 7) = dt;   // x_world 对 vx_world 的影响
    F(1, 8) = dt;   // y_world 对 vy_world 的影响
    F(2, 9) = dt;   // z_world 对 vz_world 的影响
    F(6, 10) = dt;  // heading_world 对 v_heading_world 的影响
    
    // 大地坐标系部分
    F(11, 15) = dt;  // x_earth 对 vx_earth 的影响
    F(12, 16) = dt;  // y_earth 对 vy_earth 的影响
    F(13, 17) = dt;  // z_earth 对 vz_earth 的影响
    F(14, 18) = dt;  // heading_earth 对 v_heading_earth 的影响
    
    // 2. 设置观测矩阵 H
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(11, 19);
    // 车辆坐标系观测
    H.block<7,7>(0,0) = Eigen::MatrixXd::Identity(7,7);  // x,y,z,w,l,h,heading
    // 大地坐标系观测
    H.block<4,4>(7,11) = Eigen::MatrixXd::Identity(4,4); // x,y,z,heading

    // 3. 设置过程噪声协方差矩阵 Q
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(19, 19);
    Q.block<7,7>(0,0) *= 1.0;    // 位置和尺寸噪声较小
    Q.block<4,4>(7,7) *= 10.0;   // 速度过程噪声适中
    Q.block<4,4>(11,11) *= 1.0;  // 大地坐标系位置噪声较小
    Q.block<4,4>(15,15) *= 10.0; // 大地坐标系速度噪声适中

    // 4. 设置测量噪声协方差矩阵 R
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(11, 11);
    R.block<3,3>(0,0) *= 0.1;     // 位置测量噪声小
    R.block<3,3>(3,3) *= 1.0;     // 尺寸测量噪声较大
    R(6,6) = 0.1;                 // 航向角测量噪声小
    R.block<4,4>(7,7) *= 0.1;     // 大地坐标系测量噪声小

    // 5. 设置初始状态向量
    Eigen::VectorXd x = Eigen::VectorXd::Zero(19);
    x.head<7>() = initial_pos.head<7>();  // 车辆坐标系状态
    x.segment<4>(11) = initial_pos.tail<4>();  // 大地坐标系状态
    
    // 初始化速度为小值而不是0
    x.segment<4>(7).setConstant(0.1);   // 车辆坐标系速度
    x.segment<4>(15).setConstant(0.1);  // 大地坐标系速度

    // 6. 设置初始状态协方差矩阵
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(19, 19);
    P.block<7,7>(0,0) *= 1.0;     // 位置和尺寸的初始不确定性较小
    P.block<4,4>(7,7) *= 10.0;    // 速度的初始不确定性适中
    P.block<4,4>(11,11) *= 1.0;   // 大地坐标系位置的初始不确定性较小
    P.block<4,4>(15,15) *= 10.0;  // 大地坐标系速度的初始不确定性适中

    // 重要：先设置矩阵，再设置状态
    if (auto* kf = dynamic_cast<KalmanFilter*>(filter.get())) {
        kf->set_transition_matrix(F);
        kf->set_measurement_matrix(H);
        kf->set_process_noise(Q);
        kf->set_measurement_noise(R);
        kf->set_state(x);
        kf->set_covariance(P);
    } else if (auto* ekf = dynamic_cast<ExtendedKalmanFilter*>(filter.get())) {
        ekf->set_transition_matrix(F);
        ekf->set_process_noise(Q);
        ekf->set_measurement_noise(R);
        ekf->set_state(x);
        ekf->set_covariance(P);
    } else if (auto* iekf = dynamic_cast<IteratedExtendedKalmanFilter*>(filter.get())) {
        iekf->set_transition_matrix(F);
        iekf->set_process_noise(Q);
        iekf->set_measurement_noise(R);
        iekf->set_state(x);
        iekf->set_covariance(P);
    }

    // 添加调试信息
    // std::cout << "Initial state: " << x.transpose() << std::endl;
    // std::cout << "Transition matrix F:\n" << F << std::endl;
}

// 最重要的函数
void Filter::update(const target_t& detection, float confidence) {
    // 1. 更新不需要卡尔曼滤波的属性
    info["x_pixel"] = detection.x_pixel;
    info["y_pixel"] = detection.y_pixel;
    info["w_pixel"] = detection.w_pixel;
    info["h_pixel"] = detection.h_pixel;
    info["time_stamp"] = detection.time_stamp;
    info["property"] = detection.property;
    info["k"] = detection.k;
    info["s"] = detection.s;
    
    // 更新水平面属性
    info["x_world1"] = detection.x_world1;
    info["y_world1"] = detection.y_world1;
    info["w_world1"] = detection.w_world1;
    info["h_world1"] = detection.h_world1;
    info["l_world1"] = detection.l_world1;
    
    // 更新三角形属性
    info["x_world2"] = detection.x_world2;
    info["y_world2"] = detection.y_world2;
    info["w_world2"] = detection.w_world2;
    info["h_world2"] = detection.h_world2;
    info["l_world2"] = detection.l_world2;

    // 更新分类和置信度
    info["class_id"] = detection.classid;
    info["score"] = detection.conf;

    // 保存当前航向角，用于增量更新
    previous_yaw_world = filter->get_state()(6);
    previous_yaw_earth = filter->get_state()(14);

    // 2. 保存原始点集
    points_world = detection.points_world;
    points_earth = detection.points_earth;
    last_detection = detection;

    // 3. 处理位置突变
    // 容易引起其他问题,暂时搁置

    // 4. 处理角度偏移
    auto [new_yaw_world, new_yaw_earth] = handle_heading_change(detection, confidence);

    // 5. 处理尺寸变化
    auto [final_w, final_l, final_h] = handle_size_change(detection, confidence);

    // 6. 构建观测向量
    Eigen::VectorXd z(11);
    z << detection.x_world, detection.y_world, detection.z_world,
         final_w, final_l, final_h,
         new_yaw_world,
         detection.x_earth, detection.y_earth, detection.z_earth,
         new_yaw_earth;

    // 使用增量更新航向角
    z(6) = previous_yaw_world + limit_angle(new_yaw_world - previous_yaw_world);
    z(10) = previous_yaw_earth + limit_angle(new_yaw_earth - previous_yaw_earth);

    // 7. 更新滤波器
    if (filter) {
        // 添加状态检查
        Eigen::VectorXd pre_state = filter->get_state();
        bool has_invalid = false;
        for (int i = 0; i < pre_state.size(); ++i) {
            if (!std::isfinite(pre_state(i))) {
                has_invalid = true;
                std::cout << "Warning: Invalid state detected at index " << i << ": " << pre_state(i) << std::endl;
            }
        }

        if (!has_invalid) {
            filter->update(z);
            
            // 检查更新后的状态
            Eigen::VectorXd state = filter->get_state();
            bool state_valid = true;
            for (int i = 0; i < state.size(); ++i) {
                if (!std::isfinite(state(i))) {
                    state_valid = false;
                    // std::cout << "Warning: NaN/Inf detected at index " << i << " after update" << std::endl;
                }
            }

            if (!state_valid) {
                // 如果状态无效，回退到预测状态
                filter->set_state(pre_state);
                // std::cout << "Rolling back to previous state due to invalid update" << std::endl;
            } else {
                // 正常更新，限制航向角范围
                state(6) = limit_angle(state(6));   // 车辆坐标系航向角
                state(14) = limit_angle(state(14)); // 大地坐标系航向角
                filter->set_state(state);
            }
        }
    }

    // 8. 更新跟踪器状态
    prev_confidence = confidence;
    hits++;

    track_history.push_back(Box3D(detection.x_world, detection.y_world, detection.z_world,
                                 final_w, final_l, final_h,
                                 new_yaw_world,
                                 detection.classid, detection.conf));
}

// 处理航向角变化
std::pair<double, double> Filter::handle_heading_change(
    const target_t& detection, 
    double confidence) {
    double previous_yaw_world = filter->get_state()(6);
    double new_yaw_world = detection.heading_world;
    double previous_yaw_earth = filter->get_state()(14);
    double new_yaw_earth = detection.heading_earth;
    
    double yaw_diff_world = limit_angle(new_yaw_world - previous_yaw_world);
    double yaw_diff_earth = limit_angle(new_yaw_earth - previous_yaw_earth);
    bool large_yaw_change_world = std::abs(yaw_diff_world) > M_PI / 12;
    bool large_yaw_change_earth = std::abs(yaw_diff_earth) > M_PI / 12;

    if (large_yaw_change_world) {
        if (hits < 6) {
            if (confidence > prev_confidence) {
                // 采用新的yaw
            } else if (confidence < prev_confidence) {
                new_yaw_world = previous_yaw_world;
            } else {
                new_yaw_world = (previous_yaw_world + new_yaw_world) / 2;
            }
        } else {
            new_yaw_world = previous_yaw_world;
        }
    }
        
    if (large_yaw_change_earth) {
        if (hits < 6) {
            if (confidence > prev_confidence) {
                // 采用新的yaw
            } else if (confidence < prev_confidence) {
                new_yaw_earth = previous_yaw_earth;
            } else {
                new_yaw_earth = (previous_yaw_earth + new_yaw_earth) / 2;
            }
        } else {
            new_yaw_earth = previous_yaw_earth;
        }
    }

    return {new_yaw_world, new_yaw_earth};
}

// 处理尺寸变化
std::tuple<double, double, double> Filter::handle_size_change(
    const target_t& detection,
    double confidence) {
    Eigen::VectorXd current_state = filter->get_state();
    double current_w = current_state(3);
    double current_l = current_state(4);
    double current_h = current_state(5);

    // 计算尺寸变化比例
    double w_ratio = std::abs(detection.w_world / current_w);
    double l_ratio = std::abs(detection.l_world / current_l);
    double h_ratio = std::abs(detection.h_world / current_h);

    // 设置形变阈值（允许20%的变化）
    const double shape_change_threshold = 1.2;
    bool large_shape_change = (w_ratio > shape_change_threshold || w_ratio < 1/shape_change_threshold ||
                             l_ratio > shape_change_threshold || l_ratio < 1/shape_change_threshold ||
                             h_ratio > shape_change_threshold || h_ratio < 1/shape_change_threshold);

    double final_w = detection.w_world;
    double final_l = detection.l_world;
    double final_h = detection.h_world;

    if (large_shape_change) {
        if (hits < 6) {
            if (confidence > prev_confidence) {
                // 采用新的尺寸
            } else {
                final_w = current_w;
                final_l = current_l;
                final_h = current_h;
            }
        } else {
            final_w = (current_w + detection.w_world) / 2;
            final_l = (current_l + detection.l_world) / 2;
            final_h = (current_h + detection.h_world) / 2;
        }
    }

    return {final_w, final_l, final_h};
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
            
            // 如果类别不同，设置IoU为负值，确保不会匹配
            if (boxa_3d.class_id != boxb_3d.class_id) {
                iou_matrix(d, t) = -1;
                continue;
            }
            
            auto [giou, iou3d, iou2d] = calculate_iou(boxa_3d, boxb_3d);
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
        std::cout << "Detection " << match[0] << " -> Tracker " << match[1] << std::endl;
    }
    
    std::cout << "\nUnmatched Detections:";
    for (int id : unmatched_detections) {
        std::cout << " " << id;
    }
    
    std::cout << "\nUnmatched Trackers:";
    for (int id : unmatched_trackers) {
        std::cout << " " << id;
    }
    std::cout << std::endl;
}

Eigen::VectorXd Filter::get_world_state() const {
    if (filter) {
        return filter->get_state().head<7>();
    }
    return Eigen::VectorXd::Zero(7);
}

Eigen::VectorXd Filter::get_earth_state() const {
    if (filter) {
        Eigen::VectorXd state = filter->get_state();
        Eigen::VectorXd earth_state(4);
        earth_state << state.segment<3>(11), state(14);  // x,y,z,heading
        return earth_state;
    }
    return Eigen::VectorXd::Zero(4);
}

Eigen::VectorXd Filter::get_state() const {
    if (filter) {
        return filter->get_state();
    }
    return Eigen::VectorXd::Zero(19);
}

Eigen::VectorXd Filter::get_velocity() const {
    if (filter) {
        Eigen::VectorXd state = filter->get_state();
        return state.segment<3>(7);  // vx, vy, vz in world frame
    }
    return Eigen::VectorXd::Zero(3);
}

float Filter::get_yaw_speed() const {
    if (filter) {
        Eigen::VectorXd state = filter->get_state();
        float vx = state(7);  // vx in world frame
        float vy = state(8);  // vy in world frame
        float v = std::sqrt(vx * vx + vy * vy);
        float yaw_rate = state(10);  // heading rate in world frame
        return yaw_rate;
    }
    return 0.0f;
}

const std::vector<Box3D>& Filter::get_history() const {
    return track_history;
}

std::vector<point_t> Filter::track_world_prediction(int steps) const {
    std::vector<point_t> predictions;
    if (!filter) return predictions;

    // 创建临时副本进行预测
    auto temp_filter = std::unique_ptr<BaseFilter>(filter->clone());
    Eigen::VectorXd current_state = temp_filter->get_state();

    // 保存当前位置
    point_t current_point;
    current_point.x = current_state(0);
    current_point.y = current_state(1);
    predictions.push_back(current_point);

    // 预测未来位置
    for (int i = 0; i < steps; i++) {
        temp_filter->predict();
        current_state = temp_filter->get_state();
        
        point_t point;
        point.x = current_state(0);
        point.y = current_state(1);
        predictions.push_back(point);
    }

    return predictions;
}

std::vector<point_t> Filter::track_earth_prediction(int steps) const {
    std::vector<point_t> predictions;
    if (!filter) return predictions;

    // 创建临时副本进行预测
    auto temp_filter = std::unique_ptr<BaseFilter>(filter->clone());
    Eigen::VectorXd current_state = temp_filter->get_state();

    // 保存当前位置
    point_t current_point;
    current_point.x = current_state(11);  // x in earth frame
    current_point.y = current_state(12);  // y in earth frame
    predictions.push_back(current_point);

    // 预测未来位置
    for (int i = 0; i < steps; i++) {
        temp_filter->predict();
        current_state = temp_filter->get_state();
        
        point_t point;
        point.x = current_state(11);
        point.y = current_state(12);
        predictions.push_back(point);
    }

    return predictions;
}


