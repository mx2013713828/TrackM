/*
 * File:        matching.cpp
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Implementation of assignment algorithms.
 */


#include <algorithm>
#include <numeric>
#include <iostream>
#include "../include/matching.h"
#include "../include/giou.h"

namespace Association {

MatchResult greedy_match(const std::vector<Box3D>& detections, 
                        const std::vector<Box3D>& trackers, 
                        float iou_threshold) 
{
    MatchResult result;
    if (trackers.empty()) {
        result.unmatched_detections.resize(detections.size());
        std::iota(result.unmatched_detections.begin(), result.unmatched_detections.end(), 0);
        return result;
    }

    Eigen::MatrixXf iou_matrix = Eigen::MatrixXf::Zero(detections.size(), trackers.size());

    for (size_t d = 0; d < detections.size(); ++d) {
        for (size_t t = 0; t < trackers.size(); ++t) {
            const Box3D& boxa_3d = detections[d];
            const Box3D& boxb_3d = trackers[t];
            
            // 如果类别不同，设置IoU为负值，确保不会匹配
            if (boxa_3d.class_id != boxb_3d.class_id) {
                iou_matrix(d, t) = -1.0f;
                continue;
            }
            
            // 计算相似度
            auto [giou, iou3d, iou2d] = calculate_iou_with_yaw(boxa_3d, boxb_3d);
            iou_matrix(d, t) = giou;
        }
    }

    std::vector<int> row_indices(detections.size());
    std::vector<int> col_indices(trackers.size());
    std::iota(row_indices.begin(), row_indices.end(), 0);
    std::iota(col_indices.begin(), col_indices.end(), 0);

    // 按最大IoU排序行索引，实现简单的优先级
    std::sort(row_indices.begin(), row_indices.end(), [&iou_matrix](int i1, int i2) {
        return iou_matrix.row(i1).maxCoeff() > iou_matrix.row(i2).maxCoeff();
    });

    for (int i : row_indices) {
        int best_j = -1;
        float best_iou = -1.0f;

        for (int j : col_indices) {
            if (iou_matrix(i, j) > best_iou) {
                best_iou = iou_matrix(i, j);
                best_j = j;
            }
        }

        if (best_iou >= iou_threshold) {
            result.matches.push_back({i, best_j});
            // 从可用列中移除已匹配项
            col_indices.erase(std::remove(col_indices.begin(), col_indices.end(), best_j), col_indices.end());
        } else {
            result.unmatched_detections.push_back(i);
        }
    }

    for (int j : col_indices) {
        result.unmatched_trackers.push_back(j);
    }

    return result;
}

void print_results(const MatchResult& result) 
{
    std::cout << "Matches:" << std::endl;
    for (const auto& match : result.matches) {
        std::cout << "Detection " << match[0] << " -> Tracker " << match[1] << std::endl;
    }
    
    std::cout << "\nUnmatched Detections:";
    for (int id : result.unmatched_detections) {
        std::cout << " " << id;
    }
    
    std::cout << "\nUnmatched Trackers:";
    for (int id : result.unmatched_trackers) {
        std::cout << " " << id;
    }
    std::cout << std::endl;
}

} // namespace Association
