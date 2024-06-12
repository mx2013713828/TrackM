/*
 * File:        trackm.cpp
 * Author:      Yufeng Ma
 * Date:        2024-06-01
 * Email:       97357473@qq.com
 * Description: Implementation of the tracking filter classes and matching algorithm.
 */

#include "../include/trackm.h"
#include "../include/kalman_filter.h"
#include <algorithm>
#include <iostream>
#include <tuple>
#include <Eigen/Dense>
#include "../include/giou.h"
#include <numeric>
#include <vector>
#include <unordered_map>

Filter::Filter(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int ID)
    : initial_pos(bbox3D), time_since_update(0), id(ID), hits(1), info(info) {}

KF::KF(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int ID)
    : Filter(bbox3D, info, ID), kf(10, 7) {
    _init_kalman_filter();
}

void KF::_init_kalman_filter() {
    kf.F = Eigen::MatrixXd::Identity(10, 10);
    kf.F(0, 7) = kf.F(1, 8) = kf.F(2, 9) = 1;

    kf.H = Eigen::MatrixXd::Zero(7, 10);
    for (int i = 0; i < 7; ++i) {
        kf.H(i, i) = 1;
    }
    kf.P.bottomRightCorner(3, 3) *= 1000;
    kf.P *= 10;

    kf.Q.bottomRightCorner(3, 3) *= 0.01;
    kf.x.head<7>() = initial_pos;

    
}

void KF::predict() {
    kf.predict();
}

void KF::update(const Eigen::VectorXd& bbox3D) {
    kf.update(bbox3D);
}

Eigen::VectorXd KF::get_state() const {
    return kf.x;
}

Eigen::VectorXd KF::get_velocity() const {
    return kf.x.tail<3>();
}

std::tuple<std::vector<std::array<int, 2>>, std::vector<int>, std::vector<int>>
associate_detections_to_trackers(const std::vector<Box3D>& detections,
                                 const std::vector<Box3D>& trackers,
                                 float iou_threshold) {
    if (trackers.empty()) {
        return std::make_tuple(std::vector<std::array<int, 2>>(), 
                               std::vector<int>(detections.size()), 
                               std::vector<int>());
    }

    Eigen::MatrixXf iou_matrix = Eigen::MatrixXf::Zero(detections.size(), trackers.size());

    for (size_t d = 0; d < detections.size(); ++d) {
        for (size_t t = 0; t < trackers.size(); ++t) {
            Box3D boxa_3d = detections[d];
            Box3D boxb_3d = trackers[t];
            auto [giou, iou3d, iou2d] = calculate_iou(boxa_3d, boxb_3d);
            iou_matrix(d, t) = giou;
        }
    }

    std::vector<int> row_indices(detections.size());
    std::vector<int> col_indices(trackers.size());
    std::iota(row_indices.begin(), row_indices.end(), 0);
    std::iota(col_indices.begin(), col_indices.end(), 0);

    std::sort(row_indices.begin(), row_indices.end(), [&iou_matrix](int i1, int i2) {
        return iou_matrix.row(i1).maxCoeff() > iou_matrix.row(i2).maxCoeff();
    });

    std::vector<std::array<int, 2>> matches;
    std::vector<int> unmatched_detections, unmatched_trackers;

    for (int i : row_indices) {
        int best_j = -1;
        float best_iou = -1.0;

        for (int j : col_indices) {
            if (iou_matrix(i, j) > best_iou) {
                best_iou = iou_matrix(i, j);
                best_j = j;
            }
        }

        if (best_iou >= iou_threshold) {
            matches.push_back({i, best_j});
            col_indices.erase(std::remove(col_indices.begin(), col_indices.end(), best_j), col_indices.end());
        } else {
            unmatched_detections.push_back(i);
        }
    }

    for (int j : col_indices) {
        unmatched_trackers.push_back(j);
    }

    return std::make_tuple(matches, unmatched_detections, unmatched_trackers);
}

void print_results(const std::vector<std::array<int, 2>>& matches,
                   const std::vector<int>& unmatched_detections,
                   const std::vector<int>& unmatched_trackers) {
    std::cout << "Matches:" << std::endl;
    for (const auto& match : matches) {
        std::cout << "Detection " << match[0] << " matched with Tracker " << match[1] << std::endl;
    }

    std::cout << "Unmatched Detections:" << std::endl;
    for (int idx : unmatched_detections) {
        std::cout << "Detection " << idx << std::endl;
    }

    std::cout << "Unmatched Trackers:" << std::endl;
    for (int idx : unmatched_trackers) {
        std::cout << "Tracker " << idx << std::endl;
    }
}
