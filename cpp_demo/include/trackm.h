/*
 * File:        trackm.h
 * Author:      Yufeng Ma
 * Date:        2024-06-01
 * Email:       97357473@qq.com
 * Description: Header file for the Filter and KF classes, which implement
 *              tracking filters using Kalman Filters, and functions for
 *              associating detections with trackers.
 */

#ifndef TRACKM_H
#define TRACKM_H

#include <vector>
#include <array>
#include <unordered_map>
#include <Eigen/Dense>
#include "giou.h"
// #include "kalman_filter.h" 
#include "ekf.h"

class Filter {
public:
    Eigen::VectorXd initial_pos;
    int time_since_update;
    int track_id;
    int hits;
    std::unordered_map<std::string, float> info;

    Filter(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int Track_ID);
    virtual void predict() = 0;
    virtual void update(const target_t& detection, float confidence) = 0;
};

class KF : public Filter {
public:
    KF(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int Track_ID);
    
    void predict() override;
    void update(const target_t& detection, float confidence) override;
    
    // 获取两个坐标系下的状态
    Eigen::VectorXd get_world_state() const;
    Eigen::VectorXd get_earth_state() const;
    
    // 获取两个坐标系下的预测轨迹
    std::vector<point_t> track_world_prediction(int steps) const;
    std::vector<point_t> track_earth_prediction(int steps) const;

    // 添加获取原始点集的方法
    const std::vector<point_t>& get_points_world() const { return points_world; }
    const std::vector<point_t>& get_points_earth() const { return points_earth; }

    target_t last_detection;  // 存储最后一次更新的检测数据

    // 添加缺失的函数声明
    Eigen::VectorXd get_state() const;
    Eigen::VectorXd get_velocity() const;
    float get_yaw_speed() const;
    const std::vector<Box3D>& get_history() const;

private:
    void _init_kalman_filter();
    EKalmanFilter ekf;
    std::vector<Box3D> track_history;
    float prev_confidence;
    std::vector<point_t> points_world;  // 原始车辆坐标系点集
    std::vector<point_t> points_earth;  // 原始大地坐标系点集
};


std::tuple<std::vector<std::array<int, 2>>, std::vector<int>, std::vector<int>>
associate_detections_to_trackers(const std::vector<Box3D>& detections,
                               const std::vector<Box3D>& trackers,
                               float iou_threshold = 0.1);

// 打印匹配结果,测试时使用
void print_results(const std::vector<std::array<int, 2>>& matches,
                   const std::vector<int>& unmatched_detections,
                   const std::vector<int>& unmatched_trackers) ;

#endif // TRACKM_H
