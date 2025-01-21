/*
 * File:        kalman_filter.h
 * Author:      Yufeng Ma
 * Date:        2025-01-20
 * Email:       97357473@qq.com
 * Description: Standard Kalman filter declarations.
 */

#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <Eigen/Dense>
#include "base_filter.h"

class KalmanFilter : public BaseFilter {
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

    BaseFilter* clone() const override {
        return new KalmanFilter(*this);
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

#endif // KALMAN_FILTER_H
