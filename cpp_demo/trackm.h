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
    int id;
    int hits;
    std::unordered_map<std::string, float> info;

    Filter(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int ID);
};

class KF : public Filter {
public:
    KalmanFilter kf;

    KF(const Eigen::VectorXd& bbox3D, const std::unordered_map<std::string, float>& info, int ID);
    void _init_kalman_filter();
    void predict();
    void update(const Eigen::VectorXd& bbox3D);
    Eigen::VectorXd get_state();
    Eigen::VectorXd get_velocity();
};

std::tuple<std::vector<std::array<int, 2>>, std::vector<int>, std::vector<int>>
associate_detections_to_trackers(const std::vector<Box3D>& detections,
                                 const std::vector<Box3D>& trackers,
                                 float iou_threshold = 0.3);

#endif // TRACKM_H
