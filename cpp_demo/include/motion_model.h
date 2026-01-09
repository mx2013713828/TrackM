/*
 * File:        motion_model.h
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Motion models for state estimation.
 */

#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace StateIdx {
    enum {
        X_WORLD = 0, Y_WORLD = 1, Z_WORLD = 2,
        W_WORLD = 3, L_WORLD = 4, H_WORLD = 5,
        YAW_WORLD = 6,
        VX_WORLD = 7, VY_WORLD = 8, VZ_WORLD = 9,
        YAW_RATE_WORLD = 10,
        
        X_EARTH = 11, Y_EARTH = 12, Z_EARTH = 13,
        YAW_EARTH = 14,
        VX_EARTH = 15, VY_EARTH = 16, VZ_EARTH = 17,
        YAW_RATE_EARTH = 18
    };
}

// 抽象基类
class MotionModel 
{
    
public:
    virtual ~MotionModel() = default;
    
    struct Matrices {
        Eigen::MatrixXd F;
        Eigen::MatrixXd H;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd R;
    };

    /**
     * @brief 生成该模型下的矩阵 F, H, Q, R
     * 
     * @param is_low_heading_weight 是否降低航向角权重
     * @return Matrices 包含4个矩阵的结构体
     */
    virtual Matrices generateMatrices(bool is_low_heading_weight) const = 0;

    /**
     * @brief 生成初始状态向量和协方差矩阵
     * 
     * @param initial_pos 初始检测框信息（前7位为车辆坐标系，后4位为大地坐标系）
     * @return std::pair<Eigen::VectorXd, Eigen::MatrixXd> {初始状态x, 初始协方差P}
     */
    /**
     * @brief 生成初始状态向量和协方差矩阵
     * 
     * @param initial_pos 初始检测框信息（前7位为车辆坐标系，后4位为大地坐标系）
     * @return std::pair<Eigen::VectorXd, Eigen::MatrixXd> {初始状态x, 初始协方差P}
     */
    virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> getInitialState(const Eigen::VectorXd& initial_pos) const = 0;

    /**
     * @brief 预测下一个状态 (非线性状态转移函数 f(x))
     * 
     * @param x 当前状态向量
     * @return Eigen::VectorXd 预测的下一个状态向量
     */
    virtual Eigen::VectorXd predictState(const Eigen::VectorXd& x) const = 0;

    /**
     * @brief 计算状态转移雅可比矩阵 F
     * 
     * @param x 当前状态向量
     * @return Eigen::MatrixXd 雅可比矩阵
     */
    virtual Eigen::MatrixXd calculateJacobian(const Eigen::VectorXd& x) const = 0;
};

// 线性恒速模型 (Linear Constant Velocity)
class LinearCVModel : public MotionModel 
{

public:
    LinearCVModel(double dt = 0.1) : dt_(dt) {}

    Matrices generateMatrices(bool is_low_heading_weight) const override {
        // 1. 设置状态转移矩阵 F (线性部分，用于KF或初始化)
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(19, 19);
        
        // 车辆坐标系部分
        F(StateIdx::X_WORLD, StateIdx::VX_WORLD) = dt_;
        F(StateIdx::Y_WORLD, StateIdx::VY_WORLD) = dt_;
        F(StateIdx::Z_WORLD, StateIdx::VZ_WORLD) = dt_;
        F(StateIdx::YAW_WORLD, StateIdx::YAW_RATE_WORLD) = dt_;
        F(StateIdx::YAW_RATE_WORLD, StateIdx::YAW_RATE_WORLD) = 0.5; // 阻尼项？
        
        // 大地坐标系部分
        F(StateIdx::X_EARTH, StateIdx::VX_EARTH) = dt_;
        F(StateIdx::Y_EARTH, StateIdx::VY_EARTH) = dt_;
        F(StateIdx::Z_EARTH, StateIdx::VZ_EARTH) = dt_;
        F(StateIdx::YAW_EARTH, StateIdx::YAW_RATE_EARTH) = dt_;
        F(StateIdx::YAW_RATE_EARTH, StateIdx::YAW_RATE_EARTH) = 0.5;

        // 2. 设置观测矩阵 H
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(11, 19);
        // 车辆坐标系观测
        H.block<7,7>(0,0) = Eigen::MatrixXd::Identity(7,7);  
        // 大地坐标系观测
        H.block<4,4>(7,11) = Eigen::MatrixXd::Identity(4,4); 

        // 3. 设置过程噪声协方差矩阵 Q
        Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(19, 19);
        Q(StateIdx::YAW_RATE_WORLD, StateIdx::YAW_RATE_WORLD) = 0.01;
        Q(StateIdx::YAW_RATE_EARTH, StateIdx::YAW_RATE_EARTH) = 0.01;
        
        Q.block<7,7>(0,0) *= 1.0;     // 位置和尺寸噪声较小
        Q.block<4,4>(7,7) *= 10.0;    // 速度过程噪声适中
        Q.block<4,4>(11,11) *= 1.0;   // 大地坐标系位置噪声较小
        Q.block<4,4>(15,15) *= 10.0;  // 大地坐标系速度噪声适中
        
        if (is_low_heading_weight) {
            Q(StateIdx::YAW_RATE_WORLD, StateIdx::YAW_RATE_WORLD) *= 0.1;
            Q(StateIdx::YAW_RATE_EARTH, StateIdx::YAW_RATE_EARTH) *= 0.1;
        }

        // 4. 设置测量噪声协方差矩阵 R
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(11, 11);
        R.block<3,3>(0,0) *= 0.1;     // 位置测量噪声小
        R.block<3,3>(3,3) *= 1.0;     // 尺寸测量噪声较大
        R.block<3,3>(7,7) *= 0.1;     // 大地坐标系位置测量噪声小

        if (is_low_heading_weight) {
            R(6,6) = 1000.0;           // 车辆坐标系航向角噪声很大
            R(10,10) = 1000.0;         // 大地坐标系航向角噪声很大
        } else {
            R(6,6) = 0.1;             // 车辆坐标系航向角噪声小
            R(10,10) = 0.1;           // 大地坐标系航向角噪声小
        }

        return {F, H, Q, R};
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> getInitialState(const Eigen::VectorXd& initial_pos) const override {
        // 5. 设置初始状态向量
        Eigen::VectorXd x = Eigen::VectorXd::Zero(19);
        x.head<7>() = initial_pos.head<7>();
        x.segment<4>(11) = initial_pos.tail<4>();
        
        // 初始化速度为小值而不是0
        x.segment<4>(StateIdx::VX_WORLD).setConstant(0.0);
        x.segment<4>(StateIdx::VX_EARTH).setConstant(0.0);

        // 6. 设置初始状态协方差矩阵
        Eigen::MatrixXd P = Eigen::MatrixXd::Identity(19, 19);
        P.block<7,7>(0,0) *= 1.0;
        P.block<4,4>(7,7) *= 10.0;
        P.block<4,4>(11,11) *= 1.0;
        P.block<4,4>(15,15) *= 10.0;

        return {x, P};
    }

    Eigen::VectorXd predictState(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd x_pred = x;

        // 车辆坐标系状态更新 - 非线性运动模型 (CTRV-like logic)
        double vx_world = x(StateIdx::VX_WORLD), vy_world = x(StateIdx::VY_WORLD), vz_world = x(StateIdx::VZ_WORLD);
        double heading_world = x(StateIdx::YAW_WORLD), v_heading_world = x(StateIdx::YAW_RATE_WORLD);
        
        // 更新位置
        x_pred(StateIdx::X_WORLD) += vx_world * std::cos(heading_world) * dt_ - vy_world * std::sin(heading_world) * dt_;
        x_pred(StateIdx::Y_WORLD) += vx_world * std::sin(heading_world) * dt_ + vy_world * std::cos(heading_world) * dt_;
        x_pred(StateIdx::Z_WORLD) += vz_world * dt_;
        x_pred(StateIdx::YAW_WORLD) += v_heading_world * dt_;

        // 大地坐标系状态更新 - 非线性运动模型
        double vx_earth = x(StateIdx::VX_EARTH), vy_earth = x(StateIdx::VY_EARTH), vz_earth = x(StateIdx::VZ_EARTH);
        double heading_earth = x(StateIdx::YAW_EARTH), v_heading_earth = x(StateIdx::YAW_RATE_EARTH);
        
        // 更新位置
        x_pred(StateIdx::X_EARTH) += vx_earth * std::cos(heading_earth) * dt_ - vy_earth * std::sin(heading_earth) * dt_;
        x_pred(StateIdx::Y_EARTH) += vx_earth * std::sin(heading_earth) * dt_ + vy_earth * std::cos(heading_earth) * dt_;
        x_pred(StateIdx::Z_EARTH) += vz_earth * dt_;
        x_pred(StateIdx::YAW_EARTH) += v_heading_earth * dt_;

        return x_pred;
    }

    Eigen::MatrixXd calculateJacobian(const Eigen::VectorXd& x) const override {
        Eigen::MatrixXd F_jacobian = Eigen::MatrixXd::Identity(19, 19);

        // 车辆坐标系的非线性运动模型的雅可比矩阵
        double heading_world = x(StateIdx::YAW_WORLD);
        double vx_world = x(StateIdx::VX_WORLD), vy_world = x(StateIdx::VY_WORLD);
        
        // 对 x_world 的偏导数
        F_jacobian(StateIdx::X_WORLD, StateIdx::YAW_WORLD) = (-vx_world * std::sin(heading_world) - vy_world * std::cos(heading_world)) * dt_;
        F_jacobian(StateIdx::X_WORLD, StateIdx::VX_WORLD) = std::cos(heading_world) * dt_;
        F_jacobian(StateIdx::X_WORLD, StateIdx::VY_WORLD) = -std::sin(heading_world) * dt_;
        
        // 对 y_world 的偏导数
        F_jacobian(StateIdx::Y_WORLD, StateIdx::YAW_WORLD) = (vx_world * std::cos(heading_world) - vy_world * std::sin(heading_world)) * dt_;
        F_jacobian(StateIdx::Y_WORLD, StateIdx::VX_WORLD) = std::sin(heading_world) * dt_;
        F_jacobian(StateIdx::Y_WORLD, StateIdx::VY_WORLD) = std::cos(heading_world) * dt_;
        
        // 其他线性部分保持不变
        F_jacobian(StateIdx::Z_WORLD, StateIdx::VZ_WORLD) = dt_;    // z_world 对 vz_world 的偏导数
        F_jacobian(StateIdx::YAW_WORLD, StateIdx::YAW_RATE_WORLD) = dt_;   // heading_world 对 v_heading_world 的偏导数

        // 大地坐标系部分的雅可比矩阵计算
        double heading_earth = x(StateIdx::YAW_EARTH);
        double vx_earth = x(StateIdx::VX_EARTH), vy_earth = x(StateIdx::VY_EARTH);
        
        F_jacobian(StateIdx::X_EARTH, StateIdx::YAW_EARTH) = (-vx_earth * std::sin(heading_earth) - vy_earth * std::cos(heading_earth)) * dt_;
        F_jacobian(StateIdx::X_EARTH, StateIdx::VX_EARTH) = std::cos(heading_earth) * dt_;
        F_jacobian(StateIdx::X_EARTH, StateIdx::VY_EARTH) = -std::sin(heading_earth) * dt_;
        
        F_jacobian(StateIdx::Y_EARTH, StateIdx::YAW_EARTH) = (vx_earth * std::cos(heading_earth) - vy_earth * std::sin(heading_earth)) * dt_;
        F_jacobian(StateIdx::Y_EARTH, StateIdx::VX_EARTH) = std::sin(heading_earth) * dt_;
        F_jacobian(StateIdx::Y_EARTH, StateIdx::VY_EARTH) = std::cos(heading_earth) * dt_;
        
        F_jacobian(StateIdx::Z_EARTH, StateIdx::VZ_EARTH) = dt_;  // z_earth 对 vz_earth 的偏导数
        F_jacobian(StateIdx::YAW_EARTH, StateIdx::YAW_RATE_EARTH) = dt_;  // heading_earth 对 v_heading_earth 的偏导数

        return F_jacobian;
    }

private:
    double dt_;
};
