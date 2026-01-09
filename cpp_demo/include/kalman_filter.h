/*
 * File:        kalman_filter.h
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Standard Kalman filter declarations.
 */

#pragma once

#include <Eigen/Dense>
#include "base_filter.h"

class KalmanFilter : public BaseFilter 
{

public:
    KalmanFilter(int state_dim, int measurement_dim);
    
    void predict() override;
    void update(const Eigen::VectorXd& z) override;
    
    Eigen::VectorXd get_state() const override { return x; }
    Eigen::MatrixXd get_covariance() const override { return P; }
    
    void set_state(const Eigen::VectorXd& x) override;
    void set_covariance(const Eigen::MatrixXd& P) override;
    
    // 设置模型参数
    void set_transition_matrix(const Eigen::MatrixXd& F);
    void set_measurement_matrix(const Eigen::MatrixXd& H);
    void set_process_noise(const Eigen::MatrixXd& Q);
    void set_measurement_noise(const Eigen::MatrixXd& R);

    void init(const Eigen::MatrixXd& F_in, const Eigen::MatrixXd& H_in, 
              const Eigen::MatrixXd& Q_in, const Eigen::MatrixXd& R_in, 
              const Eigen::VectorXd& x_in, const Eigen::MatrixXd& P_in,
              std::shared_ptr<MotionModel> model = nullptr) override {
        F = F_in;
        H = H_in;
        Q = Q_in;
        R = R_in;
        x = x_in;
        P = P_in;
    }

    std::unique_ptr<BaseFilter> clone() const override {
        return std::make_unique<KalmanFilter>(*this);
    }

private:
    int state_dim;
    int measurement_dim;
    Eigen::VectorXd x;  // 状态向量
    Eigen::MatrixXd F;  // 状态转移矩阵
    Eigen::MatrixXd H;  // 观测矩阵
    Eigen::MatrixXd P;  // 协方差矩阵
    Eigen::MatrixXd R;  // 测量噪声矩阵
    Eigen::MatrixXd Q;  // 过程噪声矩阵
};
