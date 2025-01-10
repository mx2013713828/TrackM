#ifndef EKALMAN_FILTER_H
#define EKALMAN_FILTER_H

#include <Eigen/Dense>

class EKalmanFilter {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EKalmanFilter(int state_dim, int measure_dim);
    
    // 公有成员变量
    Eigen::VectorXd x;  // 状态向量
    Eigen::MatrixXd F;  // 状态转移矩阵
    Eigen::MatrixXd H;  // 观测矩阵
    Eigen::MatrixXd P;  // 协方差矩阵
    Eigen::MatrixXd R;  // 观测噪声矩阵
    Eigen::MatrixXd Q;  // 过程噪声矩阵
    
    // 成员函数
    void predict();
    void update(const Eigen::VectorXd& z);
    
private:
    int state_dim;
    int measure_dim;
    
    // 添加私有成员函数声明
    Eigen::VectorXd f(const Eigen::VectorXd& x);
    Eigen::MatrixXd calculate_jacobian_f(const Eigen::VectorXd& x);
    Eigen::MatrixXd calculate_jacobian_h(const Eigen::VectorXd& x);
};
#endif // EKALMAN_FILTER_H
