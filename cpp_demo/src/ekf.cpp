/*
 * File:        kalman_filter.cpp
 * Author:      Yufeng Ma
 * Date:        2024-12-06
 * Email:       97357473@qq.com
 * Description: Implements the EKF class methods, including the
 *              constructor, predict, and update functions, using the
 *              Eigen library for matrix operations.
 */

#include "../include/ekf.h"

EKalmanFilter::EKalmanFilter(int state_dim, int measure_dim)
    : state_dim(state_dim), measure_dim(measure_dim) {
    x = Eigen::VectorXd::Zero(state_dim);
    F = Eigen::MatrixXd::Identity(state_dim, state_dim);
    H = Eigen::MatrixXd::Zero(measure_dim, state_dim);
    P = Eigen::MatrixXd::Identity(state_dim, state_dim);
    R = Eigen::MatrixXd::Identity(measure_dim, measure_dim);
    Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

Eigen::VectorXd EKalmanFilter::f(const Eigen::VectorXd& x) {
    Eigen::VectorXd x_pred = x;
    double dt = 0.1; // 时间步长

    // 状态变量提取
    double vx = x(7), vy = x(8), yaw = x(6), yaw_rate = x(10);

    // 非线性状态转移
    x_pred(0) += vx * std::cos(yaw) * dt - vy * std::sin(yaw) * dt; // x 位置更新
    x_pred(1) += vx * std::sin(yaw) * dt + vy * std::cos(yaw) * dt; // y 位置更新
    x_pred(2) += x(9) * dt;                                         // z 位置更新
    x_pred(6) += yaw_rate * dt;                                     // 偏航角更新

    return x_pred;
}

Eigen::MatrixXd EKalmanFilter::calculate_jacobian_f(const Eigen::VectorXd& x) {
    Eigen::MatrixXd F_jacobian = Eigen::MatrixXd::Identity(state_dim, state_dim);
    double dt = 0.1;
    double yaw = x(6);
    double vx = x(7), vy = x(8);

    // 偏导数计算
    F_jacobian(0, 6) = -vx * std::sin(yaw) * dt - vy * std::cos(yaw) * dt;
    F_jacobian(0, 7) = std::cos(yaw) * dt;
    F_jacobian(0, 8) = -std::sin(yaw) * dt;
    F_jacobian(1, 6) = vx * std::cos(yaw) * dt - vy * std::sin(yaw) * dt;
    F_jacobian(1, 7) = std::sin(yaw) * dt;
    F_jacobian(1, 8) = std::cos(yaw) * dt;
    F_jacobian(6, 10) = dt; // yaw 对 yaw_rate 的导数

    return F_jacobian;
}

Eigen::MatrixXd EKalmanFilter::calculate_jacobian_h(const Eigen::VectorXd& x) {
    Eigen::MatrixXd H_jacobian = Eigen::MatrixXd::Zero(measure_dim, state_dim);
    for (int i = 0; i < measure_dim; ++i) {
        H_jacobian(i, i) = 1; // 假设测量模型是直接观测 x, y, z 等
    }
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
    // 计算测量残差
    Eigen::VectorXd y = z - H * x;

    // 计算雅可比矩阵 H
    H = calculate_jacobian_h(x);

    // 卡尔曼增益计算
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();

    // 更新状态向量和误差协方差矩阵
    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * H) * P;
}

