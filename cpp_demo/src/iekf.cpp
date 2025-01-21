/*
 * File:        iekf.cpp
 * Author:      Yufeng Ma
 * Date:        2025-01-20
 * Email:       97357473@qq.com
 * Description: Iterated EKF for improved nonlinear estimation.
 */
#include "../include/iekf.h"
#include <iostream>
#include <cmath>

IteratedExtendedKalmanFilter::IteratedExtendedKalmanFilter(int state_dim, int measurement_dim)
    : state_dim(state_dim), measure_dim(measurement_dim), 
      max_iterations(10), convergence_threshold(1e-6) {
    // 初始化状态向量和矩阵
    x = Eigen::VectorXd::Zero(state_dim);
    P = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    R = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim);
    F = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

void IteratedExtendedKalmanFilter::predict() {
    // 使用非线性状态转移函数
    x = f(x);
    
    // 计算状态转移雅可比矩阵
    F = calculate_jacobian_f(x);
    
    // 更新协方差
    P = F * P * F.transpose() + Q;
}

void IteratedExtendedKalmanFilter::update(const Eigen::VectorXd& z) {
    Eigen::VectorXd x_iter = x;
    Eigen::VectorXd x_prev;
    Eigen::MatrixXd H;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        x_prev = x_iter;
        H = calculate_jacobian_h(x_iter);
        
        // 保存 K 作为成员变量
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        K = P * H.transpose() * S.inverse();
        
        x_iter = x + K * (z - H * x_iter - H * (x - x_iter));
        
        if ((x_iter - x_prev).norm() < convergence_threshold) {
            break;
        }
    }
    
    x = x_iter;
    
    // 使用最终的 H 更新协方差
    H = calculate_jacobian_h(x);  // 使用最终状态计算 H
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim, state_dim);
    P = (I - K * H) * P;
}

BaseFilter* IteratedExtendedKalmanFilter::clone() const {
    auto* clone = new IteratedExtendedKalmanFilter(state_dim, measure_dim);
    clone->x = x;
    clone->P = P;
    clone->Q = Q;
    clone->R = R;
    clone->F = F;
    clone->K = K;
    clone->max_iterations = max_iterations;
    clone->convergence_threshold = convergence_threshold;
    return clone;
}

Eigen::VectorXd IteratedExtendedKalmanFilter::f(const Eigen::VectorXd& x) {
    Eigen::VectorXd x_pred = x;
    double dt = 0.1;

    // 车辆坐标系状态更新 - 非线性运动模型
    double vx_world = x(7), vy_world = x(8), vz_world = x(9);
    double heading_world = x(6), v_heading_world = x(10);
    
    // 更新位置
    x_pred(0) += vx_world * std::cos(heading_world) * dt - vy_world * std::sin(heading_world) * dt;
    x_pred(1) += vx_world * std::sin(heading_world) * dt + vy_world * std::cos(heading_world) * dt;
    x_pred(2) += vz_world * dt;
    x_pred(6) += v_heading_world * dt;

    // 大地坐标系状态更新 - 非线性运动模型
    double vx_earth = x(15), vy_earth = x(16), vz_earth = x(17);
    double heading_earth = x(14), v_heading_earth = x(18);
    
    // 更新位置
    x_pred(11) += vx_earth * std::cos(heading_earth) * dt - vy_earth * std::sin(heading_earth) * dt;
    x_pred(12) += vx_earth * std::sin(heading_earth) * dt + vy_earth * std::cos(heading_earth) * dt;
    x_pred(13) += vz_earth * dt;
    x_pred(14) += v_heading_earth * dt;

    return x_pred;
}

Eigen::MatrixXd IteratedExtendedKalmanFilter::calculate_jacobian_f(const Eigen::VectorXd& x) {
    Eigen::MatrixXd F_jacobian = Eigen::MatrixXd::Identity(state_dim, state_dim);
    double dt = 0.1;

    // 车辆坐标系的非线性运动模型的雅可比矩阵
    double heading_world = x(6);
    double vx_world = x(7), vy_world = x(8);
    
    // 对 x_world 的偏导数
    F_jacobian(0, 6) = (-vx_world * std::sin(heading_world) - vy_world * std::cos(heading_world)) * dt;
    F_jacobian(0, 7) = std::cos(heading_world) * dt;
    F_jacobian(0, 8) = -std::sin(heading_world) * dt;
    
    // 对 y_world 的偏导数
    F_jacobian(1, 6) = (vx_world * std::cos(heading_world) - vy_world * std::sin(heading_world)) * dt;
    F_jacobian(1, 7) = std::sin(heading_world) * dt;
    F_jacobian(1, 8) = std::cos(heading_world) * dt;
    
    // 其他线性部分保持不变
    F_jacobian(2, 9) = dt;    // z_world 对 vz_world 的偏导数
    F_jacobian(6, 10) = dt;   // heading_world 对 v_heading_world 的偏导数

    // 大地坐标系部分的雅可比矩阵计算（类似车辆坐标系）
    double heading_earth = x(14);
    double vx_earth = x(15), vy_earth = x(16);
    
    F_jacobian(11, 14) = (-vx_earth * std::sin(heading_earth) - vy_earth * std::cos(heading_earth)) * dt;
    F_jacobian(11, 15) = std::cos(heading_earth) * dt;
    F_jacobian(11, 16) = -std::sin(heading_earth) * dt;
    
    F_jacobian(12, 14) = (vx_earth * std::cos(heading_earth) - vy_earth * std::sin(heading_earth)) * dt;
    F_jacobian(12, 15) = std::sin(heading_earth) * dt;
    F_jacobian(12, 16) = std::cos(heading_earth) * dt;
    
    F_jacobian(13, 17) = dt;  // z_earth 对 vz_earth 的偏导数
    F_jacobian(14, 18) = dt;  // heading_earth 对 v_heading_earth 的偏导数

    return F_jacobian;
}

Eigen::MatrixXd IteratedExtendedKalmanFilter::calculate_jacobian_h(const Eigen::VectorXd& x) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(measure_dim, state_dim);
    
    // 车辆坐标系观测 - 直接观测
    H.block<7,7>(0,0) = Eigen::MatrixXd::Identity(7,7);  // x,y,z,w,l,h,heading
    
    // 大地坐标系观测 - 直接观测
    H.block<4,4>(7,11) = Eigen::MatrixXd::Identity(4,4); // x,y,z,heading
    
    return H;
} 