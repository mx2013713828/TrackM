/*
 * File:        ekf.h
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Extended Kalman filter declarations.
 */

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <functional>
#include "base_filter.h"  // 包含基类定义

class ExtendedKalmanFilter : public BaseFilter 
{

public:
    ExtendedKalmanFilter(int state_dim, int measurement_dim);
    
    void predict() override;
    void update(const Eigen::VectorXd& z) override;
    
    Eigen::VectorXd get_state() const override { return x; }
    Eigen::MatrixXd get_covariance() const override { return P; }
    
    void set_state(const Eigen::VectorXd& x) override;
    void set_covariance(const Eigen::MatrixXd& P) override;
    
    // EKF 特有的方法
    void set_state_transition_function(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f);
    void set_measurement_function(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h);
    void set_jacobian_F(std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> jacobian_F);
    void set_jacobian_H(std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> jacobian_H);
    
    // 添加设置状态转移矩阵的方法
    void set_transition_matrix(const Eigen::MatrixXd& new_F) {
        F = new_F;
    }
    
    void set_process_noise(const Eigen::MatrixXd& new_Q) {
        Q = new_Q;
    }
    
    void set_measurement_noise(const Eigen::MatrixXd& new_R) {
        R = new_R;
    }
    
    void init(const Eigen::MatrixXd& F_in, const Eigen::MatrixXd& H_in, 
              const Eigen::MatrixXd& Q_in, const Eigen::MatrixXd& R_in, 
              const Eigen::VectorXd& x_in, const Eigen::MatrixXd& P_in,
              std::shared_ptr<MotionModel> model = nullptr) override {
        // 对于 EKF，F 是雅可比矩阵的初始值，H 在此处可能暂时用不到（如果是纯非线性）
        // 但为了统一接口，我们也可以先保存
        F = F_in; 
        Q = Q_in;
        R = R_in;
        x = x_in;
        P = P_in;
        motion_model_ = model;
    }

    std::unique_ptr<BaseFilter> clone() const override {
        return std::make_unique<ExtendedKalmanFilter>(*this);
    }

private:
    int state_dim;
    int measure_dim;
    Eigen::VectorXd x;  // 状态向量
    Eigen::MatrixXd P;  // 协方差矩阵
    Eigen::MatrixXd Q;  // 过程噪声矩阵
    Eigen::MatrixXd R;  // 测量噪声矩阵
    Eigen::MatrixXd F;  // 状态转移矩阵
    
    // 添加私有成员函数声明
    Eigen::VectorXd f(const Eigen::VectorXd& x);
    Eigen::MatrixXd calculate_jacobian_f(const Eigen::VectorXd& x);
    Eigen::MatrixXd calculate_jacobian_h(const Eigen::VectorXd& x);
};
