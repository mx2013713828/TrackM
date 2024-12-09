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
    TrackManager(int max_age, int min_hits); // 构造函数
    void update(const std::vector<Box3D>& detections); // 更新所有跟踪器
    std::vector<Box3D> get_reliable_tracks (); // 返回有效跟踪器的状态
    std::vector<KF>& get_all_trackers() {
        return trackers;
    }
private:
    void create_new_trackers(const std::vector<Box3D>& detections, const std::vector<int>& unmatched_detections); // 创建新的跟踪器
    void update_trackers(const std::vector<Box3D>& detections, const std::vector<std::array<int, 2>>& matches); // 更新命中跟踪器
    void increment_age_unmatched_trackers(const std::vector<int>& unmatched_trackers); // 增加未命中跟踪器的age


    std::vector<KF> trackers;
    int next_id;
    int max_age;
    int min_hits;
};
