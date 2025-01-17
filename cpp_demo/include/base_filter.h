#ifndef BASE_FILTER_H
#define BASE_FILTER_H

#include <Eigen/Dense>

class BaseFilter {
public:
    virtual ~BaseFilter() = default;
    
    // 纯虚函数，子类必须实现
    virtual void predict() = 0;
    virtual void update(const Eigen::VectorXd& z) = 0;
    
    // 获取状态
    virtual Eigen::VectorXd get_state() const = 0;
    virtual Eigen::MatrixXd get_covariance() const = 0;
    
    // 设置状态
    virtual void set_state(const Eigen::VectorXd& x) = 0;
    virtual void set_covariance(const Eigen::MatrixXd& P) = 0;
    
    // 添加克隆接口
    virtual BaseFilter* clone() const = 0;
};

#endif // BASE_FILTER_H 