/*
 * File:        track_manager.cpp
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Multi-object tracker management implementation.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map> 
#include <Eigen/Dense>

#include "../include/track_manager.h"
#include "../include/trackm.h"
#include "../include/kalman_filter.h"
#include "../include/giou.h"
#include "../include/matching.h"

/* ----------------------------------------------------------------- */
// 函数名: TrackManager()                   
// 说　明: 构造函数              
/* ----------------------------------------------------------------- */ 
TrackManager::TrackManager(int max_age, int min_hits)
    : next_id(0), max_age(max_age), min_hits(min_hits) 
{

}

/* ----------------------------------------------------------------- */
// 函数名: update()                   
// 说　明: 更新跟踪器 
// 输　入: 检测框列表
/* ----------------------------------------------------------------- */
void TrackManager::update(const std::vector<target_t>& detections) 
{
    // 预测所有跟踪器的状态
    for (auto& tracker : trackers) {
        tracker.predict();
    }

    // 将 target_t 转换为 Box3D 用于关联
    std::vector<Box3D> detection_boxes;
    for (const auto& det : detections) {
        Box3D box(det.x_world, det.y_world, det.z_world,
                 det.w_world, det.l_world, det.h_world,
                 det.heading_world,
                 det.classid, det.conf);
        detection_boxes.push_back(box);
    }

    // 如果没有现有的跟踪器，初始化它们
    if (trackers.empty()) {
        std::vector<int> unmatched_detections(detections.size());
        std::iota(unmatched_detections.begin(), unmatched_detections.end(), 0);
        create_new_trackers(detections, unmatched_detections);
        
    } else {
        // 获取跟踪器状态
        std::vector<Box3D> tracker_states;
        for (const auto& tracker : trackers) {
            Eigen::VectorXd state = tracker.get_world_state();
            Box3D box(state(0), state(1), state(2),  // x, y, z
                     state(3), state(4), state(5),   // w, l, h
                     state(6),                       // heading
                     static_cast<int>(tracker.info.at("class_id")),  // class_id
                     tracker.info.at("score"));      // score
            tracker_states.push_back(box);
        }

        // 关联检测和跟踪器
        Association::MatchResult match_result = 
            Association::greedy_match(detection_boxes, tracker_states, 0.2);

        // 更新跟踪器
        update_trackers(detections, match_result.matches);

        // 对未命中的检测框创建新的跟踪器
        create_new_trackers(detections, match_result.unmatched_detections);

        // 增加未命中跟踪器的age
        increment_age_unmatched_trackers(match_result.unmatched_trackers);
    }

    // 移除过期的跟踪器
    trackers.erase(
        std::remove_if(trackers.begin(), trackers.end(),
                      [this](const Track& tracker) {
                          return tracker.time_since_update > max_age;
                      }),
        trackers.end());
}

/* ----------------------------------------------------------------- */
// 函数名: distance()                   
// 说　明: 计算跟踪框距离              
// 输　出: 距离值
/* ----------------------------------------------------------------- */
float TrackManager::distance(target_t a, target_t b) const 
{
    return sqrt(pow(a.x_world - b.x_world, 2) + pow(a.y_world - b.y_world, 2) + pow(a.z_world - b.z_world, 2));
}

/* ----------------------------------------------------------------- */
// 函数名: get_reliable_tracks()                   
// 说　明: 获取可靠的跟踪器              
// 输　出: 可靠的跟踪器列表
/* ----------------------------------------------------------------- */
std::vector<target_t> TrackManager::get_reliable_tracks() const 
{
    
    std::vector<target_t> reliable_tracks;
    for (const auto& tracker : trackers) {
        if (tracker.hits >= min_hits && tracker.time_since_update < max_age) {
            target_t target;
            
            // 1. 更新车辆坐标系状态
            Eigen::VectorXd world_state = tracker.get_world_state();
            // std::cout << "World state: " << world_state.transpose() << std::endl;
            
            target.x_world = world_state(0);
            target.y_world = world_state(1);
            target.z_world = world_state(2);
            target.w_world = world_state(3);
            target.l_world = world_state(4);
            target.h_world = world_state(5);
            target.heading_world = world_state(6);
            
            // 2. 更新大地坐标系状态
            Eigen::VectorXd earth_state = tracker.get_earth_state();
            target.x_earth = earth_state(0);
            target.y_earth = earth_state(1);
            target.z_earth = earth_state(2);
            target.heading_earth = earth_state(3);
            
            // 3. 更新速度信息
            Eigen::VectorXd velocity = tracker.get_velocity();
            target.vx = velocity(0);
            target.vy = velocity(1);

            // 使用大地坐标系下的速度计算 speed
            double vx_earth = tracker.get_state()(15);  // 大地坐标系下的 vx
            double vy_earth = tracker.get_state()(16);  // 大地坐标系下的 vy
            target.speed = std::sqrt(vx_earth * vx_earth + vy_earth * vy_earth);  // 使用大地坐标系速度
            
            // 4. 更新预测轨迹
            if (target.speed >= 0){ // 速度大于v0时，更新预测轨迹
                target.points_world_predict = tracker.track_world_prediction(20);
                target.points_earth_predict = tracker.track_earth_prediction(20);
            }
            
            
            // 5. 更新跟踪相关属性
            target.track_id = tracker.track_id;
            target.frames = tracker.hits;
                
            // 6. 设置不需要卡尔曼滤波更新的属性
            target.x_pixel = tracker.info.at("x_pixel");
            target.y_pixel = tracker.info.at("y_pixel");
            target.w_pixel = tracker.info.at("w_pixel");
            target.h_pixel = tracker.info.at("h_pixel");
            target.time_stamp = tracker.info.at("time_stamp");
            target.property = tracker.info.at("property");
            target.k = tracker.info.at("k");
            target.s = tracker.info.at("s");
            target.classid = static_cast<int>(tracker.info.at("class_id"));
            target.conf = tracker.info.at("score");
            
            // 添加原始点集
            target.points_world = tracker.get_points_world();
            target.points_earth = tracker.get_points_earth();
            
            // 添加水平面属性
            target.x_world1 = tracker.info.at("x_world1");
            target.y_world1 = tracker.info.at("y_world1");
            target.w_world1 = tracker.info.at("w_world1");
            target.h_world1 = tracker.info.at("h_world1");
            target.l_world1 = tracker.info.at("l_world1");
            
            // 添加三角形属性
            target.x_world2 = tracker.info.at("x_world2");
            target.y_world2 = tracker.info.at("y_world2");
            target.w_world2 = tracker.info.at("w_world2");
            target.h_world2 = tracker.info.at("h_world2");
            target.l_world2 = tracker.info.at("l_world2");
            
            
            reliable_tracks.push_back(target);
        }
    }
    // 针对重合跟踪器去重，隐藏特定条件跟踪器
    // 初始化一个标记表
    std::vector<int> reliable_tracks_idx1(reliable_tracks.size());
    std::vector<int> reliable_tracks_idx2(reliable_tracks.size());
    std::iota(reliable_tracks_idx1.begin(), reliable_tracks_idx1.end(), 0);
    std::iota(reliable_tracks_idx2.begin(), reliable_tracks_idx2.end(), 0);

    // 初始化结果表
    std::vector<target_t> result;

    for (int idx1 : reliable_tracks_idx1) {
        float min_distance = 10000;
        int min_idx = 0;
        // 找出最小距离
        for (int idx2: reliable_tracks_idx2){
            // 类别
            int c1 = reliable_tracks[idx1].classid;
            int c2 = reliable_tracks[idx2].classid;
            if (c1 == c2) {
                // 距离
                float temp_distance = distance(reliable_tracks[idx1], reliable_tracks[idx2]);
                if (temp_distance > 0.000001 && min_distance > temp_distance) {
                    min_distance = temp_distance;
                    min_idx = idx2;
                }
            }
        }
        // 找出最小距离自身框短边长一半的对象
        float l = reliable_tracks[idx1].l_world;
        float w = reliable_tracks[idx1].w_world;
        if (min_distance < (l > w ? w/2 : l/2)) {
            // 判定新老追踪
            int old_idx, new_idx;
            if (reliable_tracks[idx1].track_id > reliable_tracks[min_idx].track_id) {
                old_idx = min_idx;
                new_idx = idx1;
            } else {
                old_idx = idx1;
                new_idx = min_idx;
            }
            // 如果老追踪此刻成立，且新追踪此刻不成立；或者新老追踪都不成立时老追踪在近期有更新
            if ((reliable_tracks[old_idx].time_since_update == 0 && reliable_tracks[new_idx].time_since_update != 0) || 
                (reliable_tracks[old_idx].time_since_update != 0 && reliable_tracks[new_idx].time_since_update != 0 &&
                reliable_tracks[old_idx].time_since_update < reliable_tracks[new_idx].time_since_update)) {
                    result.push_back(reliable_tracks[old_idx]);
            } else if (reliable_tracks[old_idx].time_since_update == 0 && reliable_tracks[new_idx].time_since_update == 0) {
                result.push_back(reliable_tracks[old_idx]);
                result.push_back(reliable_tracks[new_idx]);
            } else {
                result.push_back(reliable_tracks[new_idx]);
            }
            // 删除新老id对应追踪
            reliable_tracks_idx2.erase(std::remove(reliable_tracks_idx2.begin(), reliable_tracks_idx2.end(), old_idx), reliable_tracks_idx2.end());
            reliable_tracks_idx2.erase(std::remove(reliable_tracks_idx2.begin(), reliable_tracks_idx2.end(), new_idx), reliable_tracks_idx2.end());
        }
    }
    for (int idx : reliable_tracks_idx2) {
        result.push_back(reliable_tracks[idx]);
    }
    return result;
}
 

// 初始化
void TrackManager::create_new_trackers(const std::vector<target_t>& detections, 
                                     const std::vector<int>& unmatched_detections) 
{
    for (int idx : unmatched_detections) {
        const target_t& det = detections[idx];
        
        // 构建初始状态向量
        Eigen::VectorXd bbox3D(11);
        bbox3D << det.x_world, det.y_world, det.z_world,
                  det.w_world, det.l_world, det.h_world,
                  det.heading_world,
                  det.x_earth, det.y_earth, det.z_earth,
                  det.heading_earth;
        
        // 设置所有属性
        std::unordered_map<std::string, float> info = {
            // 基本属性
            {"score", det.conf}, 
            {"class_id", static_cast<float>(det.classid)},
            {"x_pixel", static_cast<float>(det.x_pixel)},
            {"y_pixel", static_cast<float>(det.y_pixel)},
            {"w_pixel", static_cast<float>(det.w_pixel)},
            {"h_pixel", static_cast<float>(det.h_pixel)},
            {"time_stamp", static_cast<float>(det.time_stamp)},
            {"property", static_cast<float>(det.property)},
            {"k", det.k},
            {"s", det.s},
            
            // 水平面属性
            {"x_world1", det.x_world1},
            {"y_world1", det.y_world1},
            {"w_world1", det.w_world1},
            {"h_world1", det.h_world1},
            {"l_world1", det.l_world1},
            
            // 三角形属性
            {"x_world2", det.x_world2},
            {"y_world2", det.y_world2},
            {"w_world2", det.w_world2},
            {"h_world2", det.h_world2},
            {"l_world2", det.l_world2}
        };
        
        // 创建跟踪器
        // Track tracker(bbox3D, info, next_id++, FilterType::KF);    // 线性系统使用KF
        // Track tracker(bbox3D, info, next_id++, FilterType::EKF);   // 非线性系统使用EKF
        Track tracker(bbox3D, info, next_id++, FilterType::IEKF);  // 非线性系统使用IEKF，更稳定
        
        // 设置原始点集
        tracker.points_world = det.points_world;
        tracker.points_earth = det.points_earth;
        
        trackers.push_back(std::move(tracker));
    }
}

void TrackManager::update_trackers(const std::vector<target_t>& detections, 
                                 const std::vector<std::array<int, 2>>& matches) 
{
    for (const auto& match : matches) {
        int detection_idx = match[0];
        int tracker_idx = match[1];
        const target_t& det = detections[detection_idx];
        
        trackers[tracker_idx].update(det, det.conf);
        
        trackers[tracker_idx].hits++;
        trackers[tracker_idx].time_since_update = 0;
    }
}

void TrackManager::increment_age_unmatched_trackers(const std::vector<int>& unmatched_trackers) 
{
    for (int idx : unmatched_trackers) {
        trackers[idx].time_since_update++;
    }
}