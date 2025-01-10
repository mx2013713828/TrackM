#ifndef EKALMAN_FILTER_H
#define EKALMAN_FILTER_H

#include <Eigen/Dense>

class EKalmanFilter {
public:
    EKalmanFilter(int state_dim, int measure_dim);
    
    // 状态向量定义 (扩展为包含两个坐标系的状态)
    // [0-6]:   x_world, y_world, z_world, w, l, h, heading
    // [7-9]:   vx_world, vy_world, vz_world
    // [10]:    v_heading_world
    // [11-13]: x_earth, y_earth, z_earth
    // [14]:    heading_earth
    // [15-17]: vx_earth, vy_earth, vz_earth
    // [18]:    v_heading_earth
    
    void predict();
    void update(const Eigen::VectorXd& z);
    
private:
    Eigen::VectorXd f(const Eigen::VectorXd& x);
    Eigen::MatrixXd calculate_jacobian_f(const Eigen::VectorXd& x);
    Eigen::MatrixXd calculate_jacobian_h(const Eigen::VectorXd& x);
};
#endif // EKALMAN_FILTER_H
