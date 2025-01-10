/*
 * File:        track_manager.h
 * Author:      Yufeng Ma
 * Date:        2024-06-01
 * Email:       97357473@qq.com
 * Description: This header file defines the TrackManager class, which manages 
 *              multiple object trackers using the Kalman filter. The main 
 *              functionalities include:
 *              - update: Updates all trackers with new detections.
 *              - get_reliable_tracks: Returns the states of active trackers.
 *              - create_new_trackers: Initializes new trackers for unmatched detections.
 *              - update_trackers: Updates trackers that match new detections.
 *              - increment_age_unmatched_trackers: Increases the age of unmatched trackers.
 *              The class integrates with other modules such as trackm, kalman_filter, and giou.
 */

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
    std::vector<KF>& get_all_trackers() {
        return trackers;
    }

private:
    std::vector<KF> trackers;
    int next_id;
    int max_age;
    int min_hits;

    void create_new_trackers(const std::vector<target_t>& detections, 
                           const std::vector<int>& unmatched_detections);
    void update_trackers(const std::vector<target_t>& detections, 
                        const std::vector<std::array<int, 2>>& matches);
    void increment_age_unmatched_trackers(const std::vector<int>& unmatched_trackers);
};
