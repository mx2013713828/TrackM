/*
 * File:        kalman_filter.cpp
 * Author:      Yufeng Ma
 * Date:        2024-12-06
 * Email:       97357473@qq.com
 * Description: Implements the EKF class methods, including the
 *              constructor, predict, and update functions, using the
 *              Eigen library for matrix operations.
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

EKalmanFilter::EKalmanFilter(int state_dim, int measure_dim)
    : state_dim(state_dim), measure_dim(measure_dim) {
    x = Eigen::VectorXd::Zero(state_dim);  // x,y,z,w,l,h,yaw,vx,vy,vz,vyaw
    F = Eigen::MatrixXd::Identity(state_dim, state_dim);
    H = Eigen::MatrixXd::Zero(measure_dim, state_dim);
    P = Eigen::MatrixXd::Identity(state_dim, state_dim);
    R = Eigen::MatrixXd::Identity(measure_dim, measure_dim);
    Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

Eigen::VectorXd EKalmanFilter::f(const Eigen::VectorXd& x) {
    Eigen::VectorXd x_pred = x;
    double dt = 0.1;

    // 车辆坐标系状态更新 - 非线性运动模型
    double vx_world = x(7), vy_world = x(8), heading_world = x(6);
    x_pred(0) += vx_world * std::cos(heading_world) * dt - vy_world * std::sin(heading_world) * dt;
    x_pred(1) += vx_world * std::sin(heading_world) * dt + vy_world * std::cos(heading_world) * dt;
    x_pred(2) += x(9) * dt;
    x_pred(6) += x(10) * dt;

    // 大地坐标系状态更新 - 非线性运动模型
    double vx_earth = x(15), vy_earth = x(16), heading_earth = x(14);
    x_pred(11) += vx_earth * std::cos(heading_earth) * dt - vy_earth * std::sin(heading_earth) * dt;
    x_pred(12) += vx_earth * std::sin(heading_earth) * dt + vy_earth * std::cos(heading_earth) * dt;
    x_pred(13) += x(17) * dt;
    x_pred(14) += x(18) * dt;

    return x_pred;
}

Eigen::MatrixXd EKalmanFilter::calculate_jacobian_f(const Eigen::VectorXd& x) {
    Eigen::MatrixXd F_jacobian = Eigen::MatrixXd::Identity(state_dim, state_dim);
    double dt = 0.1;

    // 车辆坐标系的非线性运动模型的雅可比矩阵
    double heading_world = x(6);
    double vx_world = x(7), vy_world = x(8);
    F_jacobian(0, 6) = -vx_world * std::sin(heading_world) * dt - vy_world * std::cos(heading_world) * dt;
    F_jacobian(0, 7) = std::cos(heading_world) * dt;
    F_jacobian(0, 8) = -std::sin(heading_world) * dt;
    F_jacobian(1, 6) = vx_world * std::cos(heading_world) * dt - vy_world * std::sin(heading_world) * dt;
    F_jacobian(1, 7) = std::sin(heading_world) * dt;
    F_jacobian(1, 8) = std::cos(heading_world) * dt;
    F_jacobian(6, 10) = dt;

    // 大地坐标系的非线性运动模型的雅可比矩阵
    double heading_earth = x(14);
    double vx_earth = x(15), vy_earth = x(16);
    F_jacobian(11, 14) = -vx_earth * std::sin(heading_earth) * dt - vy_earth * std::cos(heading_earth) * dt;
    F_jacobian(11, 15) = std::cos(heading_earth) * dt;
    F_jacobian(11, 16) = -std::sin(heading_earth) * dt;
    F_jacobian(12, 14) = vx_earth * std::cos(heading_earth) * dt - vy_earth * std::sin(heading_earth) * dt;
    F_jacobian(12, 15) = std::sin(heading_earth) * dt;
    F_jacobian(12, 16) = std::cos(heading_earth) * dt;
    F_jacobian(14, 18) = dt;

    return F_jacobian;
}   

Eigen::MatrixXd EKalmanFilter::calculate_jacobian_h(const Eigen::VectorXd& x) {
    Eigen::MatrixXd H_jacobian = Eigen::MatrixXd::Zero(measure_dim, state_dim); // 11x19

    // 车辆坐标系观测
    H_jacobian(0, 0) = 1;  // x_world -> x_world
    H_jacobian(1, 1) = 1;  // y_world -> y_world
    H_jacobian(2, 2) = 1;  // z_world -> z_world
    H_jacobian(3, 3) = 1;  // w -> w
    H_jacobian(4, 4) = 1;  // l -> l
    H_jacobian(5, 5) = 1;  // h -> h
    H_jacobian(6, 6) = 1;  // heading_world -> heading_world

    // 大地坐标系观测
    H_jacobian(7, 11) = 1;  // x_earth -> x_earth
    H_jacobian(8, 12) = 1;  // y_earth -> y_earth
    H_jacobian(9, 13) = 1;  // z_earth -> z_earth
    H_jacobian(10, 14) = 1; // heading_earth -> heading_earth
    
    return H_jacobian;
}

void EKalmanFilter::predict() {
    // 使用非线性状态转移函数
    x = f(x);

    // 计算雅可比矩阵 F
    F = calculate_jacobian_f(x);

    // 更新误差协方差矩阵
    P = F * P * F.transpose() + Q;
}

void EKalmanFilter::update(const Eigen::VectorXd& z) {
    // 打印输入的观测向量
    // std::cout << "Observation z: " << z.transpose() << std::endl;
    // std::cout << "Current state x: " << x.transpose() << std::endl;

    // 计算测量残差
    Eigen::VectorXd y = z - H * x;
    // std::cout << "Measurement residual y: " << y.transpose() << std::endl;

    // 计算雅可比矩阵 H
    H = calculate_jacobian_h(x);
    // std::cout << "H matrix:\n" << H << std::endl;

    // 卡尔曼增益计算
    Eigen::MatrixXd PHt = P * H.transpose();
    Eigen::MatrixXd S = H * PHt + R;
    // std::cout << "Innovation covariance S:\n" << S << std::endl;

    // 检查 S 矩阵是否可逆
    Eigen::FullPivLU<Eigen::MatrixXd> lu(S);
    if (!lu.isInvertible()) {
        std::cout << "Warning: S matrix is not invertible!" << std::endl;
        return;  // 如果不可逆，直接返回
    }

    Eigen::MatrixXd K = PHt * S.inverse();
    // std::cout << "Kalman gain K:\n" << K << std::endl;

    // 更新状态向量和误差协方差矩阵
    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * H) * P;

    // std::cout << "Updated state x: " << x.transpose() << std::endl;
}

