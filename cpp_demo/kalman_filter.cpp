#include "kalman_filter.h"


//     卡尔曼滤波器基类 
//     使用eigen库实现卡尔曼滤波的相关计算
//     update
//     predict

KalmanFilter::KalmanFilter(int state_dim, int measure_dim)
    : state_dim(state_dim), measure_dim(measure_dim) {
    x = Eigen::VectorXd::Zero(state_dim);
    F = Eigen::MatrixXd::Identity(state_dim, state_dim);
    H = Eigen::MatrixXd::Zero(measure_dim, state_dim);
    P = Eigen::MatrixXd::Identity(state_dim, state_dim);
    R = Eigen::MatrixXd::Identity(measure_dim, measure_dim);
    Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
}

void KalmanFilter::predict() {
    x = F * x;
    P = F * P * F.transpose() + Q;
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    Eigen::VectorXd y = z - H * x;
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();
    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * H) * P;
}
