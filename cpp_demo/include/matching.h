/*
 * File:        matching.h
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Assignment algorithms and association logic.
 */

#pragma once

#include <vector>
#include <array>
#include <tuple>
#include "common_types.h"

namespace Association {

// 关联结果结构
struct MatchResult {
    std::vector<std::array<int, 2>> matches;
    std::vector<int> unmatched_detections;
    std::vector<int> unmatched_trackers;
};

/**
 * @brief 使用贪心算法关联检测框和跟踪器
 * 
 * @param detections 检测框集合
 * @param trackers 跟踪器（预测框）集合
 * @param iou_threshold IoU 阈值
 * @return MatchResult 关联结果
 */
MatchResult greedy_match(const std::vector<Box3D>& detections, 
                        const std::vector<Box3D>& trackers, 
                        float iou_threshold = 0.1);

void print_results(const MatchResult& result);

} // namespace Association
