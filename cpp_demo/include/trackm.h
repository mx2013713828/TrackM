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
#include "kalman_filter.h" 

class Filter {
public:
    Eigen::VectorXd initial_pos;
    int time_since_update;
    int track_id;
    int hits;
    std::unordered_map<std::string, float> info;

    Filter(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int Track_ID);
};

class KF : public Filter {
public:
    KalmanFilter kf;
    std::vector<Box3D> track_history;  // 用于记录每次的 bbox

    float prev_confidence = 0;  // 类内变量，存储前一帧的置信度

    KF(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int Track_ID);
    void _init_kalman_filter();
    void predict();
    // void update(const Eigen::VectorXd& bbox3D);
    void update(const Eigen::VectorXd& bbox3D, float confidence);  // 新增 confidence 作为输入

    Eigen::VectorXd get_state() const; // 这里声明为 const
    Eigen::VectorXd get_velocity() const;
    const std::vector<Box3D>& get_history() const;  // 新增历史记录方法

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
