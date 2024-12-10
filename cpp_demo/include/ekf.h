#ifndef EKALMAN_FILTER_H
#define EKALMAN_FILTER_H

#include <Eigen/Dense>

class EKalmanFilter {
public:
    EKalmanFilter(int state_dim, int measure_dim);

    void predict();
    void update(const Eigen::VectorXd& z);
    Eigen::VectorXd f(const Eigen::VectorXd& x); // 非线性状态转移函数
    Eigen::MatrixXd calculate_jacobian_f(const Eigen::VectorXd& x); // 状态转移函数的雅可比矩阵
    Eigen::MatrixXd calculate_jacobian_h(const Eigen::VectorXd& x); // 测量函数的雅可比矩阵

    Eigen::VectorXd x; // 状态向量
    Eigen::MatrixXd F; // 状态转移矩阵 (用作预测的线性化雅可比矩阵)
    Eigen::MatrixXd H; // 测量矩阵 (用作测量的线性化雅可比矩阵)
    Eigen::MatrixXd P; // 误差协方差矩阵
    Eigen::MatrixXd R; // 测量噪声协方差矩阵
    Eigen::MatrixXd Q; // 过程噪声协方差矩阵

private:
    int state_dim;
    int measure_dim;
};
#endif // EKALMAN_FILTER_H
