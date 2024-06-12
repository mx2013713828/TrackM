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
#include "../include/trackm.h"
#include "../include/kalman_filter.h"
#include "../include/giou.h"
#include "../include/track_manager.h"

TrackManager::TrackManager(int max_age, int min_hits)
    : next_id(0), max_age(max_age), min_hits(min_hits) {

    }

void TrackManager::update(const std::vector<Box3D>& detections) {
    std::cout<<"迭代update函数"<<std::endl;
    // 预测所有跟踪器的当前状态
    for (auto& tracker : trackers) {
        tracker.predict();
    }

    // 如果没有现有的跟踪器，初始化它们
    if (trackers.empty()) {
        std::cout<< "初始化跟踪器" <<std::endl;
        std::vector<int> unmatched_detections(detections.size());
        std::iota(unmatched_detections.begin(), unmatched_detections.end(), 0); // 创建一个从0开始的索引数组
        create_new_trackers(detections, unmatched_detections);
    } else {
        // 获取所有跟踪器的当前状态
        std::vector<Box3D> tracker_states;
        for (const auto& tracker : trackers) {
            Eigen::VectorXd state = tracker.get_state();
            tracker_states.emplace_back(state);
        }

        // 将检测结果与跟踪器关联

        auto [matches, unmatched_detections, unmatched_trackers] = associate_detections_to_trackers(detections, tracker_states);
        // 打印 matches
        std::cout << "Matches:" << std::endl;
        for (const auto& match : matches) {
            std::cout << "(" << match[0] << ", " << match[1] << ")" << std::endl;
        }

        // 打印 unmatched_detections
        std::cout << "Unmatched Detections:" << std::endl;
        for (const auto& detection : unmatched_detections) {
            std::cout << detection << std::endl;
        }

        // 打印 unmatched_trackers
        std::cout << "Unmatched Trackers:" << std::endl;
        for (const auto& tracker : unmatched_trackers) {
            std::cout << tracker << std::endl;
        }

        // 更新命中的跟踪器
        update_trackers(detections, matches);

        // 对未命中的检测框创建新的跟踪器
        create_new_trackers(detections, unmatched_detections);

        // 增加未命中跟踪器的age
        increment_age_unmatched_trackers(unmatched_trackers);
    }

    // 移除长时间未命中的跟踪器
    trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
                                  [this](const KF& tracker) { return tracker.time_since_update >= max_age; }),
                   trackers.end());
}

std::vector<Box3D> TrackManager::get_tracks() {
    std::vector<Box3D> reliable_tracks;
    for (const auto& tracker : trackers) {
        // std::cout<<"time_since_update: " <<tracker.time_since_update <<std::endl;
        if (tracker.hits >= min_hits) {
            reliable_tracks.emplace_back(tracker.get_state(), tracker.info.at("class_id"), tracker.info.at("score"), tracker.id);
        }
    }
    return reliable_tracks;
}


void TrackManager::create_new_trackers(const std::vector<Box3D>& detections, const std::vector<int>& unmatched_detections) {
    for (int idx : unmatched_detections) {
        const Box3D& det = detections[idx];
        Eigen::VectorXd bbox3D(7);
        bbox3D << det.x, det.y, det.z, det.w, det.l, det.h, det.yaw;
        std::unordered_map<std::string, float> info = {{"score", det.score}, {"class_id", static_cast<float>(det.class_id)}};
        trackers.emplace_back(bbox3D, info, next_id++);
    }
}

void TrackManager::update_trackers(const std::vector<Box3D>& detections, const std::vector<std::array<int, 2>>& matches) {
    for (const auto& match : matches) {
        int detection_idx = match[0];
        int tracker_idx = match[1];
        const Box3D& det = detections[detection_idx];
        Eigen::VectorXd bbox3D(7);
        bbox3D << det.x, det.y, det.z, det.w, det.l, det.h, det.yaw;
        trackers[tracker_idx].update(bbox3D);
        trackers[tracker_idx].hits++;
        trackers[tracker_idx].time_since_update = 0;
        trackers[tracker_idx].info["score"] = det.score;
        trackers[tracker_idx].info["class_id"] = static_cast<float>(det.class_id);
    }
}

void TrackManager::increment_age_unmatched_trackers(const std::vector<int>& unmatched_trackers) {
    for (int idx : unmatched_trackers) {
        // std::cout<<"增加为匹配上的跟踪的年龄前 trackers["<<idx<<"].time_since_update:" <<trackers[idx].time_since_update <<" track_id: "<<trackers[idx].id<<std::endl;
        trackers[idx].time_since_update++;
        // std::cout<<"增加为匹配上的跟踪的年龄后 trackers["<<idx<<"].time_since_update:" <<trackers[idx].time_since_update <<" track_id: "<<trackers[idx].id<<std::endl;
    }
}