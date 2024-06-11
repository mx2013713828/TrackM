#include <iostream>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include "./include/trackm.h"
#include "./include/track_manager.h"
int main() {
    // 创建 TrackManager 对象
    TrackManager track_manager(3, 1);

    // 创建一些测试 Box3D 对象
    std::vector<Box3D> detections;

    Eigen::VectorXd bbox1(7);
    bbox1 << 1, 1, 1, 2, 2, 2, 0.5;
    detections.emplace_back(bbox1, 1, 0.9);

    Eigen::VectorXd bbox2(7);
    bbox2 << 2, 2, 2, 2, 2, 2, 0.5;
    detections.emplace_back(bbox2, 2, 0.8);

    Eigen::VectorXd bbox3(7);
    bbox3 << 3, 3, 3, 2, 2, 2, 0.5;
    detections.emplace_back(bbox3, 3, 0.7);

    // 更新 TrackManager
    std::cout << "第一次更新:" << std::endl;
    track_manager.update(detections);

    // 打印跟踪结果
    std::vector<Box3D> tracks = track_manager.get_tracks();
    for (const auto& track : tracks) {
        std::cout << " | Position: (" << track.x << ", " << track.y << ", " << track.z << ") \n" <<" | Dim: ("<< \
        track.w<< ", "<<track.l<<", "<<track.h<<")" <<" yaw: ("<<track.yaw<<") Class_id: ("<<track.class_id <<") class_score: ("<<track.score<<") Track_id: " << track.track_id << std::endl;
    }

    // 创建新的检测结果并更新 TrackManager
    std::vector<Box3D> new_detections;

    Eigen::VectorXd bbox4(7);
    bbox4 << 1.1, 1.1, 1.0, 2, 2, 2, 0.5;
    new_detections.emplace_back(bbox4, 1, 0.91);

    Eigen::VectorXd bbox5(7);
    bbox5 << 2.1, 2.1, 2.0, 2, 2, 2, 0.5;
    new_detections.emplace_back(bbox5, 2, 0.81);

    Eigen::VectorXd bbox6(7);
    bbox6 << 3.1, 3.1, 3, 2, 2, 2, 0.5;
    new_detections.emplace_back(bbox6, 4, 0.71);

    std::cout << "第二次更新:" << std::endl;
    track_manager.update(new_detections);

    // 打印更新后的跟踪结果
    tracks = track_manager.get_tracks();
    for (const auto& track : tracks) {
        std::cout << " | Position: (" << track.x << ", " << track.y << ", " << track.z << ") \n" <<" | Dim: ("<< \
        track.w<< ", "<<track.l<<", "<<track.h<<")" <<" yaw: ("<<track.yaw<<") Class_id: ("<<track.class_id <<") class_score: ("<<track.score<<") Track_id: " << track.track_id << std::endl;
    }

    return 0;
}