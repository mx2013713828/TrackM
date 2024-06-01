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

int main() {

    // 使用 Eigen::VectorXd 初始化 Box3D
    Eigen::VectorXd bbox(7);
    bbox << 13.70101796, 4.57136452, -0.74235851, 4.433886, 1.823255, 2.0, 0.54469167;  // x, y, z, w, l, h, yaw

    Box3D eigen_box(bbox, 1, 0.9);

    std::unordered_map<std::string, float> info = {{"score", eigen_box.score}, {"class_id", eigen_box.id}};
    int track_id = 1;

    // 初始化KF对象
    KF kf(bbox, info, track_id);   

    std::cout << "Initial state: " << kf.get_state().transpose() << std::endl;

    // 进行一次预测
    kf.predict();
    std::cout << "Predicted state: " << kf.get_state().transpose() << std::endl;

    // 模拟一个新的测量
    Eigen::VectorXd new_measurement(7);
    // 13.87061676, 4.66908187, -0.64779444, 4.433886, 1.823255, 2., 0.55076867
    new_measurement << 13.8770, 4.66908, -0.64779, 4.433886, 1.823255, 2.0, 0.55076867;

    // 更新KF对象
    kf.update(new_measurement);
    std::cout << "Updated state: " << kf.get_state().transpose() << std::endl;

    // 打印速度
    std::cout << "Velocity: " << kf.get_velocity().transpose() << std::endl;

    return 0;
}

