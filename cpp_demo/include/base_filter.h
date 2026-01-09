/*
 * File:        base_filter.h
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Abstract base class for Kalman filter variants.
 */

#pragma once

#include <memory>
#include <Eigen/Dense>
#include "motion_model.h"


class BaseFilter 
{
    
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
    
    // 统一初始化接口
    virtual void init(const Eigen::MatrixXd& F, const Eigen::MatrixXd& H, 
                     const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, 
                     const Eigen::VectorXd& x, const Eigen::MatrixXd& P,
                     std::shared_ptr<MotionModel> model = nullptr) = 0;

    // 使用 std::unique_ptr 进行克隆，确保内存安全
    virtual std::unique_ptr<BaseFilter> clone() const = 0;

protected:
    std::shared_ptr<MotionModel> motion_model_;
};