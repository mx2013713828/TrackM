/*
 * File:        ekf.cpp
 * Author:      Yufeng Ma
 * Date:        2025-01-20
 * Email:       97357473@qq.com
 * Description: Extended Kalman filter for nonlinear systems.
 * 
 * 状态向量 x (19维):
 *   [0-6]:   车辆坐标系位姿 (x_world, y_world, z_world, w, l, h, heading_world)
 *   [7-10]:  车辆坐标系速度 (vx_world, vy_world, vz_world, v_heading_world)
 *   [11-14]: 大地坐标系位姿 (x_earth, y_earth, z_earth, heading_earth)
 *   [15-18]: 大地坐标系速度 (vx_earth, vy_earth, vz_earth, v_heading_earth)
 * 
 * 观测向量 z (11维):
 *   [0-6]:  车辆坐标系观测 (x_world, y_world, z_world, w, l, h, heading_world)
 *   [7-10]: 大地坐标系观测 (x_earth, y_earth, z_earth, heading_earth)
 * 
 * 主要矩阵维度:
 *   - F: 19x19 状态转移矩阵
 *   - H: 11x19 观测矩阵
 *   - P: 19x19 状态协方差矩阵
 *   - Q: 19x19 过程噪声协方差矩阵
 *   - R: 11x11 观测噪声协方差矩阵
 */

#include "../include/ekf.h"
#include <iostream>

ExtendedKalmanFilter::ExtendedKalmanFilter(int state_dim, int measurement_dim)
    : state_dim(state_dim), measure_dim(measurement_dim) {
    // 初始化状态向量和矩阵
    x = Eigen::VectorXd::Zero(state_dim);
    P = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    R = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim);
    F = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

void ExtendedKalmanFilter::predict() {
    // 使用非线性状态转移函数
    x = f(x);

    // 计算雅可比矩阵 F
    F = calculate_jacobian_f(x);

    // 更新误差协方差矩阵
    P = F * P * F.transpose() + Q;
}

void ExtendedKalmanFilter::update(const Eigen::VectorXd& z) {
    // 计算测量残差
    Eigen::MatrixXd H = calculate_jacobian_h(x);
    Eigen::VectorXd y = z - H * x;

    // 卡尔曼增益计算
    Eigen::MatrixXd PHt = P * H.transpose();
    Eigen::MatrixXd S = H * PHt + R;

    // 检查 S 矩阵是否可逆
    Eigen::FullPivLU<Eigen::MatrixXd> lu(S);
    if (!lu.isInvertible()) {
        std::cout << "Warning: S matrix is not invertible!" << std::endl;
        return;
    }

    Eigen::MatrixXd K = PHt * S.inverse();

    // 更新状态向量和误差协方差矩阵
    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * H) * P;
}

void ExtendedKalmanFilter::set_state(const Eigen::VectorXd& new_x) {
    x = new_x;
}

void ExtendedKalmanFilter::set_covariance(const Eigen::MatrixXd& new_P) {
    P = new_P;
}

Eigen::VectorXd ExtendedKalmanFilter::f(const Eigen::VectorXd& x) {
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

Eigen::MatrixXd ExtendedKalmanFilter::calculate_jacobian_f(const Eigen::VectorXd& x) {
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
    
    // z_world 和 heading_world 的线性部分
    F_jacobian(2, 9) = dt;    // z_world 对 vz_world 的偏导数
    F_jacobian(6, 10) = dt;   // heading_world 对 v_heading_world 的偏导数

    // 大地坐标系的非线性运动模型的雅可比矩阵
    double heading_earth = x(14);
    double vx_earth = x(15), vy_earth = x(16);
    
    // 对 x_earth 的偏导数
    F_jacobian(11, 14) = (-vx_earth * std::sin(heading_earth) - vy_earth * std::cos(heading_earth)) * dt;
    F_jacobian(11, 15) = std::cos(heading_earth) * dt;
    F_jacobian(11, 16) = -std::sin(heading_earth) * dt;
    
    // 对 y_earth 的偏导数
    F_jacobian(12, 14) = (vx_earth * std::cos(heading_earth) - vy_earth * std::sin(heading_earth)) * dt;
    F_jacobian(12, 15) = std::sin(heading_earth) * dt;
    F_jacobian(12, 16) = std::cos(heading_earth) * dt;
    
    // z_earth 和 heading_earth 的线性部分
    F_jacobian(13, 17) = dt;  // z_earth 对 vz_earth 的偏导数
    F_jacobian(14, 18) = dt;  // heading_earth 对 v_heading_earth 的偏导数

    return F_jacobian;
}

Eigen::MatrixXd ExtendedKalmanFilter::calculate_jacobian_h(const Eigen::VectorXd& x) {
    Eigen::MatrixXd H_jacobian = Eigen::MatrixXd::Zero(measure_dim, state_dim);

    // 车辆坐标系观测 - 直接观测
    H_jacobian.block<7,7>(0,0) = Eigen::MatrixXd::Identity(7,7);  // x,y,z,w,l,h,heading

    // 大地坐标系观测 - 直接观测
    H_jacobian.block<4,4>(7,11) = Eigen::MatrixXd::Identity(4,4); // x,y,z,heading
    
    return H_jacobian;
}

