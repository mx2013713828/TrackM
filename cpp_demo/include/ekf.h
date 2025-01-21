/*
 * File:        ekf.h
 * Author:      Yufeng Ma
 * Date:        2025-01-20
 * Email:       97357473@qq.com
 * Description: Extended Kalman filter declarations.
 */

#ifndef EKALMAN_FILTER_H
#define EKALMAN_FILTER_H

#include <Eigen/Dense>
#include <functional>
#include "base_filter.h"  // 包含基类定义

class ExtendedKalmanFilter : public BaseFilter {
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
    
    BaseFilter* clone() const override {
        return new ExtendedKalmanFilter(*this);
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
#endif // EKALMAN_FILTER_H
