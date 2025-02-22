/*
 * File:        trackm.h
 * Author:      Yufeng Ma
 * Date:        2025-01-20
 * Email:       97357473@qq.com
 * Description: Core tracking filter classes and detection-tracker association.
 */

#ifndef TRACKM_H
#define TRACKM_H

#include <memory>
#include <vector>
#include <array>
#include <unordered_map>
#include <Eigen/Dense>
#include "giou.h"
#include "base_filter.h"
#include "ekf.h"

enum class FilterType 
{
    KF,
    EKF,
    IEKF
};

class Filter 
{
public:
    Eigen::VectorXd initial_pos;
    int time_since_update;
    int track_id;
    int hits;
    std::unordered_map<std::string, float> info;
    std::vector<point_t> points_world;  // 原始车辆坐标系点集
    std::vector<point_t> points_earth;  // 原始大地坐标系点集
    target_t last_detection;  // 存储最后一次更新的检测数据

    Filter(const Eigen::VectorXd& bbox3D, 
           const std::unordered_map<std::string, float>& info, 
           int Track_ID,
           FilterType filter_type = FilterType::EKF);

    virtual void predict() { if(filter) filter->predict();}

    virtual void update(const target_t& detection, float confidence);
    
    // 获取状态
    virtual Eigen::VectorXd get_world_state() const;
    virtual Eigen::VectorXd get_earth_state() const;
    virtual Eigen::VectorXd get_state() const;
    virtual Eigen::VectorXd get_velocity() const;
    virtual float get_yaw_speed() const;
    
    // 获取轨迹
    virtual std::vector<point_t> track_world_prediction(int steps) const;
    virtual std::vector<point_t> track_earth_prediction(int steps) const;
    virtual const std::vector<Box3D>& get_history() const;
    
    // 获取点集
    const std::vector<point_t>& get_points_world() const { return points_world; }
    const std::vector<point_t>& get_points_earth() const { return points_earth; }

protected:
    std::unique_ptr<BaseFilter> filter;
    float prev_confidence;
    std::vector<Box3D> track_history;

private:
    void init_filter(FilterType filter_type);
    void _init_kalman_filter();

    // 处理航向角变化
    std::pair<double, double> handle_heading_change(const target_t& detection, double confidence);

    // 处理尺寸变化
    std::tuple<double, double, double> handle_size_change(const target_t& detection, double confidence);

    // 在类的私有成员中添加航向角记录
    double previous_yaw_world = 0.0;
    double previous_yaw_earth = 0.0;
    bool is_low_heading_weight;  // 标记是否降低航向角权重
};

// KF 类现在只是 Filter 的一个别名
using KF = Filter;

// 关联函数声明
std::tuple<std::vector<std::array<int, 2>>, std::vector<int>, std::vector<int>>
associate_detections_to_trackers(const std::vector<Box3D>& detections,
                               const std::vector<Box3D>& trackers,
                               float iou_threshold = 0.1);

void print_results(const std::vector<std::array<int, 2>>& matches,
                  const std::vector<int>& unmatched_detections,
                  const std::vector<int>& unmatched_trackers);

#endif // TRACKM_H
