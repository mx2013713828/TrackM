/*
 * File:        trackm.cpp
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Core tracking filter implementation and state estimation.
 */


#include <algorithm>
#include <iostream>
#include <numeric>  

#include "../include/trackm.h"
#include "../include/kalman_filter.h"
#include "../include/ekf.h"
#include "../include/iekf.h"
#include "../include/motion_model.h"

// 将角度限制在 [-pi, pi] 范围内
double limit_angle(double angle) 
{
    if (!std::isfinite(angle)) return angle;
    // 使用 atan2 归一化到 [-PI, PI]
    return std::atan2(std::sin(angle), std::cos(angle));
}

Track::Track(const Eigen::VectorXd& bbox3D, 
              const std::unordered_map<std::string, float>& info, 
              int Track_ID,
              FilterType filter_type)
    : initial_pos(bbox3D), time_since_update(0), track_id(Track_ID), 
      hits(1), info(info), prev_confidence(0.0) 
{
    is_low_heading_weight = (info.find("class_id") != info.end() && (info.at("class_id") == static_cast<int>(LIDAR_DET_TYPE::PEOPLE) || info.at("class_id") == static_cast<int>(LIDAR_DET_TYPE::CONE)));
    init_filter(filter_type);
}

void Track::init_filter(FilterType filter_type) 
{
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
    


void Track::_init_kalman_filter() 
{
    // 1. 初始化运动模型 (这里使用线性恒速模型)
    // 未来可以根据 class_id 或其他参数选择不同的模型
    // 使用 std::shared_ptr 管理 MotionModel
    std::shared_ptr<MotionModel> motion_model = std::make_shared<LinearCVModel>(0.1);

    // 2. 获取模型矩阵
    auto matrices = motion_model->generateMatrices(is_low_heading_weight);

    // 3. 获取初始状态
    auto initial_state = motion_model->getInitialState(initial_pos);

    // 4. 统一初始化滤波器
    if (filter) {
        filter->init(matrices.F, matrices.H, matrices.Q, matrices.R, 
                    initial_state.first, initial_state.second, motion_model);
    }
}


// 最重要的函数
void Track::update(const target_t& detection, float confidence) 
{
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
                
                // 限制角速度不超过1 rad/s
                // const double MAX_ANGULAR_VELOCITY = 1.17;  // 最大角速度限制 (rad/s)
                // state(10) = std::clamp(state(10), -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY);  // 限制车辆坐标系角速度
                // state(18) = std::clamp(state(18), -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY);  // 限制大地坐标系角速度
                
                filter->set_state(state);
            }
        }
    }

    // 8. 更新跟踪器状态
    prev_confidence = confidence;
    // hits++;

    track_history.push_back(Box3D(detection.x_world, detection.y_world, detection.z_world,
                                 final_w, final_l, final_h,
                                 new_yaw_world,
                                 detection.classid, detection.conf));
}

/* ----------------------------------------------------------------- */
// 函数名: handle_heading_change()                   
// 说　明: 处理航向角变化              
/* ----------------------------------------------------------------- */
std::pair<double, double> Track::handle_heading_change(const target_t& detection, double confidence) 
{
    double previous_yaw_world = filter->get_state()(6);
    double new_yaw_world = detection.heading_world;
    double previous_yaw_earth = filter->get_state()(14);
    double new_yaw_earth = detection.heading_earth;
    
    double yaw_diff_world = limit_angle(new_yaw_world - previous_yaw_world);
    double yaw_diff_earth = limit_angle(new_yaw_earth - previous_yaw_earth);
    bool large_yaw_change_world = std::abs(yaw_diff_world) > M_PI / 15;
    bool large_yaw_change_earth = std::abs(yaw_diff_earth) > M_PI / 15;
    const double alpha = 0.6;  // 平滑因子，值越小平滑效果越强
    const int stable_hits = 20; // 稳定跟踪次数

    if (large_yaw_change_world) {
        if (hits < stable_hits) {
            if (confidence > prev_confidence+0.1) {
                // 采用新的yaw
                new_yaw_world = previous_yaw_world * 0.1 + new_yaw_world * 0.9;

            } else {
                new_yaw_world = previous_yaw_world * 0.9 + new_yaw_world * 0.1;
            }
        } else {
            std::cout << "large_yaw_change_world" << ",yaw_diff_world: " <<yaw_diff_world<<std::endl;

            // new_yaw_world = previous_yaw_world + alpha * yaw_diff_world;

            if (yaw_diff_world > M_PI*0.1) {
                yaw_diff_world = M_PI*0.1;
            
            } else if (yaw_diff_world < -M_PI*0.1) {
                yaw_diff_world = -M_PI*0.1;
            }
            std::cout<<"trackid: "<<track_id<<", yaw_diff: "<<yaw_diff_world<<std::endl;
            new_yaw_world = previous_yaw_world + alpha * yaw_diff_world; // 平滑处理
        }
    } else {
        // not large_yaw_change_world
        new_yaw_world = previous_yaw_world +  yaw_diff_world;
        std::cout<<"trackid: "<<track_id<<", effective_diff: "<<yaw_diff_world<<" alpha_adapt: "<<alpha<<std::endl;
    }
        
    if (large_yaw_change_earth) {
        if (hits < stable_hits) {
            if (confidence > prev_confidence+0.1) {
                // 采用新的yaw
                new_yaw_earth = previous_yaw_earth * 0.1 + new_yaw_earth * 0.9;

            } else {
                new_yaw_earth = previous_yaw_earth * 0.9 + new_yaw_earth * 0.1;
            }
        } else {
            // new_yaw_earth = previous_yaw_earth + alpha * yaw_diff_earth;
            if (yaw_diff_earth > M_PI*0.1) {
                yaw_diff_earth = M_PI*0.1;
            
            } else if (yaw_diff_earth < -M_PI*0.1) {
                yaw_diff_earth = -M_PI*0.1;
            }
            new_yaw_earth = previous_yaw_earth + alpha * yaw_diff_earth;
        }
    } else {
        // not large_yaw_change_earth
        new_yaw_earth = previous_yaw_earth + yaw_diff_earth;
        // std::cout<<"trackid: "<<track_id<<", effective_diff: "<<yaw_diff_earth<<" alpha_adapt: "<<alpha<<std::endl;
    }

    return {new_yaw_world, new_yaw_earth};
}

// 处理尺寸变化
std::tuple<double, double, double> Track::handle_size_change(const target_t& detection,double confidence) 
{
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


Eigen::VectorXd Track::get_world_state() const 
{
    if (filter) {
        return filter->get_state().head<7>();
    }
    return Eigen::VectorXd::Zero(7);
}

Eigen::VectorXd Track::get_earth_state() const 
{
    if (filter) {
        Eigen::VectorXd state = filter->get_state();
        Eigen::VectorXd earth_state(4);
        earth_state << state.segment<3>(11), state(14);  // x,y,z,heading
        return earth_state;
    }
    return Eigen::VectorXd::Zero(4);
}

Eigen::VectorXd Track::get_state() const 
{
    if (filter) {
        return filter->get_state();
    }
    return Eigen::VectorXd::Zero(19);
}

Eigen::VectorXd Track::get_velocity() const 
{
    if (filter) {
        Eigen::VectorXd state = filter->get_state();
        return state.segment<3>(7);  // vx, vy, vz in world frame
    }
    return Eigen::VectorXd::Zero(3);
}

float Track::get_yaw_speed() const 
{
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

const std::vector<Box3D>& Track::get_history() const 
{
    return track_history;
}

std::vector<point_t> Track::track_world_prediction(int steps) const 
{
    std::vector<point_t> predictions;
    if (!filter) return predictions;

    // 创建临时副本进行预测
    auto temp_filter = filter->clone();
    Eigen::VectorXd current_state = temp_filter->get_state();

    // 保存当前位置
    point_t current_point;
    current_point.x = current_state(0);
    current_point.y = current_state(1);
    current_point.z = current_state(6);  // 添加航向角

    
    predictions.push_back(current_point);

    // 预测未来位置
    for (int i = 0; i < steps; i++) {
        temp_filter->predict();
        current_state = temp_filter->get_state();
        
        point_t point;
        point.x = current_state(0);
        point.y = current_state(1);
        point.z = current_state(6);
        predictions.push_back(point);
    }

    return predictions;
}

std::vector<point_t> Track::track_earth_prediction(int steps) const 
{
    std::vector<point_t> predictions;
    if (!filter) return predictions;

    // 创建临时副本进行预测
    auto temp_filter = filter->clone();
    Eigen::VectorXd current_state = temp_filter->get_state();

    // 保存当前位置
    point_t current_point;
    current_point.x = current_state(11);  // x in earth frame
    current_point.y = current_state(12);  // y in earth frame
    current_point.z = current_state(14);  // heading in earth frame
    
    predictions.push_back(current_point);

    // 预测未来位置
    for (int i = 0; i < steps; i++) {
        temp_filter->predict();
        current_state = temp_filter->get_state();
        
        point_t point;
        point.x = current_state(11);
        point.y = current_state(12);
        point.z = current_state(14);

        predictions.push_back(point);
    }

    return predictions;
}


