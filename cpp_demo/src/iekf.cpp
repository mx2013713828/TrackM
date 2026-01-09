/*
 * File:        iekf.cpp
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Iterated EKF for improved nonlinear estimation.
 */

#include <iostream>
#include <cmath>
#include "../include/iekf.h"

IteratedExtendedKalmanFilter::IteratedExtendedKalmanFilter(int state_dim, int measurement_dim)
    : state_dim(state_dim), measure_dim(measurement_dim), 
      max_iterations(10), convergence_threshold(1e-6) 
{
    // 初始化状态向量和矩阵
    x = Eigen::VectorXd::Zero(state_dim);
    P = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    R = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim);
    F = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

void IteratedExtendedKalmanFilter::predict() 
{
    if (motion_model_) {
        // 使用 MotionModel 进行预测
        x = motion_model_->predictState(x);
        F = motion_model_->calculateJacobian(x);
    } else {
        // Fallback: 如果没有设置 MotionModel
        x = F * x;
    }

    // 更新协方差
    P = F * P * F.transpose() + Q;
}

void IteratedExtendedKalmanFilter::update(const Eigen::VectorXd& z) 
{
    Eigen::VectorXd x_iter = x;                // 初始化迭代状态为当前状态估计
    Eigen::VectorXd x_prev;                    // 声明变量用于存储上一次迭代的状态
    Eigen::MatrixXd H;                         // 声明观测模型的雅可比矩阵
    
    for (int iter = 0; iter < max_iterations; ++iter) {  // 开始迭代循环，最多迭代max_iterations次
        x_prev = x_iter;                       // 保存当前迭代状态，用于后续收敛判断
        H = calculate_jacobian_h(x_iter);      // 在当前迭代点计算观测模型的雅可比矩阵
        
        // 保存 K 作为成员变量
        Eigen::MatrixXd S = H * P * H.transpose() + R;  // 计算创新协方差矩阵
        
        // 使用更稳定的 LDLT 分解求解线性方程组 K = P * H^T * S^-1 => K * S = P * H^T
        K = S.ldlt().solve(H * P).transpose();
        
        x_iter = x + K * (z - H * x_iter - H * (x - x_iter));  // 更新状态估计，使用迭代EKF公式
        
        if (!x_iter.allFinite() || (x_iter - x_prev).norm() < convergence_threshold) {  // 检查是否达到收敛条件
            break;                             // 如果收敛或发散，提前结束迭代
        }
    }
    
    x = x_iter;                                // 将最终迭代结果赋值给状态估计
    
    // 使用最终的 H 更新协方差
    H = calculate_jacobian_h(x);               // 使用最终状态计算观测雅可比矩阵
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim, state_dim);  // 创建单位矩阵
    P = (I - K * H) * P;                       // 更新状态协方差矩阵
}

std::unique_ptr<BaseFilter> IteratedExtendedKalmanFilter::clone() const 
{
    auto clone_ptr = std::make_unique<IteratedExtendedKalmanFilter>(state_dim, measure_dim);
    clone_ptr->x = x;
    clone_ptr->P = P;
    clone_ptr->Q = Q;
    clone_ptr->R = R;
    clone_ptr->F = F;
    clone_ptr->K = K;
    clone_ptr->max_iterations = max_iterations;
    clone_ptr->convergence_threshold = convergence_threshold;
    // CRITICAL FIX: explicit copy of motion_model_;
    // Using friend/protected access since we are in the same class scope
    clone_ptr->motion_model_ = this->motion_model_;
    return clone_ptr;
}


Eigen::MatrixXd IteratedExtendedKalmanFilter::calculate_jacobian_h(const Eigen::VectorXd& x) 
{
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(measure_dim, state_dim);
    
    // 车辆坐标系观测 - 直接观测
    H.block<7,7>(0,0) = Eigen::MatrixXd::Identity(7,7);  // x,y,z,w,l,h,heading
    
    // 大地坐标系观测 - 直接观测
    H.block<4,4>(7,11) = Eigen::MatrixXd::Identity(4,4); // x,y,z,heading
    
    return H;
} 