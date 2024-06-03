// #include <iostream>
// #include "giou.h"
// #include "trackm.h"
// int main() {
//     // 定义两个3D边界框
//     Box3D box1 = {0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f, 1.7f}; // x, y, z, w, l, h, yaw
//     Box3D box2 = {1.0f, 1.0f, 0.0f, 2.0f, 2.0f, 2.0f, 1.7f}; // x, y, z, w, l, h, yaw

//     // 计算IoU
//     auto iou_result = calculate_iou(box1, box2);

//     // 输出结果
//     std::cout << "GIoU: " << iou_result[0] << std::endl;
//     std::cout << "3D IoU: " << iou_result[1] << std::endl;
//     std::cout << "2D IoU: " << iou_result[2] << std::endl;

//     return 0;
// }
#include <iostream>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include "trackm.h"

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

int main() {
    // 测试组1：完全匹配
    std::vector<Box3D> detections1 = {
        Box3D(Eigen::VectorXd::Map(std::vector<double>{0, 0, 0, 1, 1, 1, 0}.data(), 7)),
        Box3D(Eigen::VectorXd::Map(std::vector<double>{1, 1, 1, 1, 1, 1, 0}.data(), 7))
    };
    std::vector<Box3D> trackers1 = {
        Box3D(Eigen::VectorXd::Map(std::vector<double>{0, 0, 0, 1, 1, 1, 0}.data(), 7)),
        Box3D(Eigen::VectorXd::Map(std::vector<double>{1, 1, 1, 1, 1, 1, 0}.data(), 7))
    };

    auto [matches1, unmatched_detections1, unmatched_trackers1] = associate_detections_to_trackers(detections1, trackers1, 0.3);
    std::cout << "Test Group 1: Complete Match" << std::endl;
    print_results(matches1, unmatched_detections1, unmatched_trackers1);
    std::cout << std::endl;

    // 测试组2：部分匹配
    std::vector<Box3D> detections2 = {
        Box3D(Eigen::VectorXd::Map(std::vector<double>{0, 0, 0, 1, 1, 1, 0}.data(), 7)),
        Box3D(Eigen::VectorXd::Map(std::vector<double>{1, 1, 1, 1, 1, 1, 0}.data(), 7)),
        Box3D(Eigen::VectorXd::Map(std::vector<double>{2, 2, 2, 1, 1, 1, 0}.data(), 7))
    };
    std::vector<Box3D> trackers2 = {
        Box3D(Eigen::VectorXd::Map(std::vector<double>{0, 0, 0, 1, 1, 1, 0}.data(), 7)),
        Box3D(Eigen::VectorXd::Map(std::vector<double>{2, 2, 2, 1, 1, 1, 0}.data(), 7))
    };

    auto [matches2, unmatched_detections2, unmatched_trackers2] = associate_detections_to_trackers(detections2, trackers2, 0.3);
    std::cout << "Test Group 2: Partial Match" << std::endl;
    print_results(matches2, unmatched_detections2, unmatched_trackers2);
    std::cout << std::endl;

    // 测试组3：不匹配
    std::vector<Box3D> detections3 = {
        Box3D(Eigen::VectorXd::Map(std::vector<double>{0, 0, 0, 1, 1, 1, 0}.data(), 7)),
        Box3D(Eigen::VectorXd::Map(std::vector<double>{1, 1, 1, 1, 1, 1, 0}.data(), 7))
    };
    std::vector<Box3D> trackers3 = {
        Box3D(Eigen::VectorXd::Map(std::vector<double>{3, 3, 3, 1, 1, 1, 0}.data(), 7)),
        Box3D(Eigen::VectorXd::Map(std::vector<double>{4, 4, 4, 1, 1, 1, 0}.data(), 7))
    };

    auto [matches3, unmatched_detections3, unmatched_trackers3] = associate_detections_to_trackers(detections3, trackers3, 0.3);
    std::cout << "Test Group 3: No Match" << std::endl;
    print_results(matches3, unmatched_detections3, unmatched_trackers3);
    std::cout << std::endl;

    return 0;
}