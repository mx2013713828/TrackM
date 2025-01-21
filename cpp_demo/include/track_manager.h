/*
 * File:        track_manager.h
 * Author:      Yufeng Ma
 * Date:        2025-01-20
 * Email:       97357473@qq.com
 * Description: Multi-object tracker management system.
 */
#pragma once

#include <iostream>
#include <Eigen/Dense>

#include "trackm.h"
#include "kalman_filter.h"
#include "giou.h"

// 函数声明

class TrackManager {
public:
    TrackManager(int max_age = 3, int min_hits = 3);
    
    void update(const std::vector<target_t>& detections);
    
    // 获取可靠的跟踪结果（满足min_hits条件的跟踪器）
    std::vector<target_t> get_reliable_tracks() const;
    
    // 获取所有跟踪器的引用（用于测试和可视化）
    std::vector<Filter>& get_all_trackers() {
        return trackers;
    }

private:
    std::vector<Filter> trackers;
    int next_id;
    int max_age;
    int min_hits;

    void create_new_trackers(const std::vector<target_t>& detections, 
                           const std::vector<int>& unmatched_detections);
    void update_trackers(const std::vector<target_t>& detections, 
                        const std::vector<std::array<int, 2>>& matches);
    void increment_age_unmatched_trackers(const std::vector<int>& unmatched_trackers);
};
