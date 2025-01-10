/*
 * File:        track_manager.cpp
 * Author:      Yufeng Ma
 * Date:        2024-06-01
 * Email:       97357473@qq.com
 * Description: Implementation of the TrackManager class, including methods for
 *              updating, creating, and managing object trackers using Kalman filters.
 */

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <numeric>
#include <vector>
#include <unordered_map> 
#include "../include/trackm.h"
#include "../include/kalman_filter.h"
#include "../include/giou.h"
#include "../include/track_manager.h"

TrackManager::TrackManager(int max_age, int min_hits)
    : next_id(0), max_age(max_age), min_hits(min_hits) {

    }

void TrackManager::update(const std::vector<target_t>& detections) {
    // 预测所有跟踪器的状态
    for (auto& tracker : trackers) {
        tracker.predict();
    }

    // 获取跟踪器状态（使用车辆坐标系状态进行关联）
    std::vector<Box3D> tracker_states;
    for (const auto& tracker : trackers) {
        tracker_states.push_back(Box3D(tracker.get_world_state()));
    }

    // 关联检测和跟踪器
    auto [matches, unmatched_detections, unmatched_trackers] = 
        associate_detections_to_trackers(detections, tracker_states);

    // 更新跟踪器
    update_trackers(detections, matches);
}

std::vector<target_t> TrackManager::get_tracks() const {
    std::vector<target_t> tracks;
    for (const auto& tracker : trackers) {
        if (tracker.hits >= min_hits && tracker.time_since_update < max_age) {
            target_t target;
            
            // 获取车辆坐标系状态和预测
            Eigen::VectorXd world_state = tracker.get_world_state();
            target.points_world_predict = tracker.track_world_prediction(20);
            
            // 获取大地坐标系状态和预测
            Eigen::VectorXd earth_state = tracker.get_earth_state();
            target.points_earth_predict = tracker.track_earth_prediction(20);
            
            // 设置其他属性...
            tracks.push_back(target);
        }
    }
    return tracks;
}
 

// 初始化时，将第一帧的检测加入历史轨迹中
void TrackManager::create_new_trackers(const std::vector<Box3D>& detections, const std::vector<int>& unmatched_detections) {
    for (int idx : unmatched_detections) {
        const Box3D& det = detections[idx];
        Eigen::VectorXd bbox3D(7);
        bbox3D << det.x, det.y, det.z, det.w, det.l, det.h, det.yaw;
        std::unordered_map<std::string, float> info = {{"score", det.score}, {"class_id", static_cast<float>(det.class_id)}};
        trackers.emplace_back(bbox3D, info, next_id++);
    }
}

// 添加track_history 保存历史轨迹
void TrackManager::update_trackers(const std::vector<Box3D>& detections, const std::vector<std::array<int, 2>>& matches) {
    for (const auto& match : matches) {
        int detection_idx = match[0];
        int tracker_idx = match[1];
        const Box3D& det = detections[detection_idx];
        Eigen::VectorXd bbox3D(7);
        bbox3D << det.x, det.y, det.z, det.w, det.l, det.h, det.yaw;
        trackers[tracker_idx].update(bbox3D, det.score);
        trackers[tracker_idx].hits++;
        trackers[tracker_idx].time_since_update = 0;
        trackers[tracker_idx].info["score"] = det.score;
        trackers[tracker_idx].info["class_id"] = static_cast<float>(det.class_id);
        // 将当前检测结果转化为 Box3D，并保存到 track_history 中
        // Eigen::VectorXd track_box3d(7);
        // track_box3d = trackers[tracker_idx].get_state();
        Box3D updated_box(trackers[tracker_idx].get_state(), det.class_id, det.score, trackers[tracker_idx].track_id, trackers[tracker_idx].get_yaw_speed());

        // Box3D updated_box(bbox3D, det.class_id, det.score, trackers[tracker_idx].track_id);
        // trackers[tracker_idx].track_history.push_back(updated_box);
    }
}

void TrackManager::increment_age_unmatched_trackers(const std::vector<int>& unmatched_trackers) {
    for (int idx : unmatched_trackers) {
        trackers[idx].time_since_update++;
    }
}