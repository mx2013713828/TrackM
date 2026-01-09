/*
 * File:        ekf.cpp
 * Author:      Yufeng Ma
 * Date:        2026-01-09
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
 
#include <iostream>
#include "../include/ekf.h"

ExtendedKalmanFilter::ExtendedKalmanFilter(int state_dim, int measurement_dim)
    : state_dim(state_dim), measure_dim(measurement_dim) 
{
    // 初始化状态向量和矩阵
    x = Eigen::VectorXd::Zero(state_dim);
    P = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    R = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim);
    F = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

void ExtendedKalmanFilter::predict() 
{
    if (motion_model_) {
        // 使用 MotionModel 进行预测
        x = motion_model_->predictState(x);
        F = motion_model_->calculateJacobian(x);
    } else {
        // Fallback: 如果没有设置 MotionModel (虽然不应该发生)
        // 使用默认的线性预测 (如果 F 有效)
        x = F * x;
    }

    // 更新误差协方差矩阵
    P = F * P * F.transpose() + Q;
}

void ExtendedKalmanFilter::update(const Eigen::VectorXd& z) 
{
    // 计算测量残差
    Eigen::MatrixXd H = calculate_jacobian_h(x);
    Eigen::VectorXd y = z - H * x;

    // 卡尔曼增益计算
    Eigen::MatrixXd PHt = P * H.transpose();
    Eigen::MatrixXd S = H * PHt + R;

    // 使用更稳定的 LDLT 分解求解线性方程组 K = P * H^T * S^-1 => K * S = P * H^T
    Eigen::MatrixXd K = S.ldlt().solve(PHt).transpose();

    // 更新状态向量和误差协方差矩阵
    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * H) * P;
}

void ExtendedKalmanFilter::set_state(const Eigen::VectorXd& new_x) 
{
    x = new_x;
}

void ExtendedKalmanFilter::set_covariance(const Eigen::MatrixXd& new_P) 
{
    P = new_P;
}


Eigen::MatrixXd ExtendedKalmanFilter::calculate_jacobian_h(const Eigen::VectorXd& x) 
{
    Eigen::MatrixXd H_jacobian = Eigen::MatrixXd::Zero(measure_dim, state_dim);

    // 车辆坐标系观测 - 直接观测
    H_jacobian.block<7,7>(0,0) = Eigen::MatrixXd::Identity(7,7);  // x,y,z,w,l,h,heading

    // 大地坐标系观测 - 直接观测
    H_jacobian.block<4,4>(7,11) = Eigen::MatrixXd::Identity(4,4); // x,y,z,heading
    
    return H_jacobian;
}

