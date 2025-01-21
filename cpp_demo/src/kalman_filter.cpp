/*
 * File:        kalman_filter.cpp
 * Author:      Yufeng Ma
 * Date:        2025-01-20
 * Email:       97357473@qq.com
 * Description: Standard Kalman filter state estimation.
 */

#include "../include/kalman_filter.h"

KalmanFilter::KalmanFilter(int state_dim, int measurement_dim) 
    : state_dim(state_dim), measurement_dim(measurement_dim) {
    // 初始化状态向量和矩阵
    x = Eigen::VectorXd::Zero(state_dim);
    F = Eigen::MatrixXd::Identity(state_dim, state_dim);
    H = Eigen::MatrixXd::Zero(measurement_dim, state_dim);
    P = Eigen::MatrixXd::Identity(state_dim, state_dim);
    R = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim);
    Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

void KalmanFilter::predict() {
    x = F * x;
    P = F * P * F.transpose() + Q;
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    Eigen::MatrixXd y = z - H * x;
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();

    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * H) * P;
}

void KalmanFilter::set_state(const Eigen::VectorXd& new_x) {
    x = new_x;
}

void KalmanFilter::set_covariance(const Eigen::MatrixXd& new_P) {
    P = new_P;
}

void KalmanFilter::set_transition_matrix(const Eigen::MatrixXd& new_F) {
    F = new_F;
}

void KalmanFilter::set_measurement_matrix(const Eigen::MatrixXd& new_H) {
    H = new_H;
}

void KalmanFilter::set_process_noise(const Eigen::MatrixXd& new_Q) {
    Q = new_Q;
}

void KalmanFilter::set_measurement_noise(const Eigen::MatrixXd& new_R) {
    R = new_R;
}
