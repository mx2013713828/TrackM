#pragma once
#include "base_filter.h"

class IteratedExtendedKalmanFilter : public BaseFilter {
public:
    IteratedExtendedKalmanFilter(int state_dim, int measurement_dim);
    void predict() override;
    void update(const Eigen::VectorXd& z) override;
    BaseFilter* clone() const override;
    
    // 添加这些必要的虚函数实现
    Eigen::VectorXd get_state() const override { return x; }
    Eigen::MatrixXd get_covariance() const override { return P; }
    void set_state(const Eigen::VectorXd& new_x) override { x = new_x; }
    void set_covariance(const Eigen::MatrixXd& new_P) override { P = new_P; }

    // 添加其他必要的函数
    void set_process_noise(const Eigen::MatrixXd& new_Q) { Q = new_Q; }
    void set_measurement_noise(const Eigen::MatrixXd& new_R) { R = new_R; }
    void set_transition_matrix(const Eigen::MatrixXd& new_F) { F = new_F; }

private:
    // 添加成员变量
    Eigen::VectorXd x;  // 状态向量
    Eigen::MatrixXd P;  // 状态协方差
    Eigen::MatrixXd Q;  // 过程噪声
    Eigen::MatrixXd R;  // 测量噪声
    Eigen::MatrixXd F;  // 状态转移矩阵
    Eigen::MatrixXd K;  // 卡尔曼增益矩阵，用于更新协方差
    
    int state_dim;      // 状态维度
    int measure_dim;    // 测量维度
    int max_iterations;  // 最大迭代次数
    double convergence_threshold;  // 收敛阈值
    
    // IEKF 特有的方法
    Eigen::VectorXd f(const Eigen::VectorXd& x);  // 非线性状态转移函数
    Eigen::MatrixXd calculate_jacobian_f(const Eigen::VectorXd& x);  // 状态转移雅可比矩阵
    Eigen::MatrixXd calculate_jacobian_h(const Eigen::VectorXd& x);  // 观测雅可比矩阵
}; 