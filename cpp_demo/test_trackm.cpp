#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <dirent.h>
#include <cstring> // 添加头文件
#include <sstream> // 添加这行
#include <Eigen/Dense>
#include "./include/giou.h"
#include "./include/kalman_filter.h"
#include "./include/trackm.h"
#include "./include/track_manager.h"

bool compareFileNames(const std::string& a, const std::string& b) {
    return std::stoi(a) < std::stoi(b);
}

int main() {
    std::string folderPath = "../data/detections/20241202_test56/"; // 替换为你的文件夹路径
    std::vector<std::string> filePaths;

    // 打开目录
    DIR *dir = opendir(folderPath.c_str());
    if (dir == nullptr) {
        std::cerr << "Failed to open directory!" << std::endl;
        return 1;
    }

    // 读取目录中的文件
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        // 忽略当前目录和父目录
        if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        // 收集文件路径
        filePaths.push_back(entry->d_name);
    }

    // 关闭目录
    closedir(dir);

    // 对文件路径进行排序
    std::sort(filePaths.begin(), filePaths.end(), compareFileNames);

    // 创建 TrackManager 对象
    TrackManager track_manager(5, 3);

    int n = 0;
    // 逐个读取每个文件的内容并按行输出
    for (const auto& filePath : filePaths) {
        std::ifstream file(folderPath + "/" + filePath); // 打开文件
        if (file.is_open()) {
            std::string line;
            std::cout << "File: " << filePath << std::endl;
            std::vector<Box3D> new_detections;

            while (std::getline(file, line)) {

                Eigen::VectorXd bbox(9);

                std::istringstream iss(line);
                if (!(iss >> bbox(0) >> bbox(1) >> bbox(2) >> bbox(3) >> bbox(4) >> bbox(5) >> bbox(6) >> bbox(7) >> bbox(8))) {
                    std::cerr << "Error parsing line: " << line << std::endl;
                    continue;
                }
                // 输出 Eigen::VectorXd 中的数据 
                // std::cout << "bbox: " << bbox.transpose() << std::endl;
                // Box3D box(bbox,bbox[7],bbox[8]);
                new_detections.emplace_back(bbox,bbox[7],bbox[8]);
                
                // for(auto &box3d:new_detections){
                //     std::cout << "box3d: " << box3d.x << " " << box3d.y << " " << box3d.z << " " << box3d.w << " " << box3d.l << " " << box3d.h << " " << box3d.yaw << " " << box3d.score << std::endl;
                // }
            }
        track_manager.update(new_detections);
        std::cout << std::endl;
        file.close(); // 关闭文件

        std::vector<Box3D> tracks;
        tracks = track_manager.get_reliable_tracks();
    
        std::ofstream savefile("../data/cpp_result/" + filePath);
        std::ofstream savefile_future("../data/cpp_result_future/" + filePath);

        // 保存txt结果
        // track type: Box3D 
        for (const auto& track : tracks) {
            std::cout << " | Position: (" << track.x << ", " << track.y << ", " << track.z << ") \n" <<" | Dim: ("<< \
            track.w<< ", "<<track.l<<", "<<track.h<<")" <<" yaw: ("<<track.yaw<<") Class_id: ("<<track.class_id <<") class_score: ("<<track.score<<") Track_id: (" << track.track_id << ") v_yaw: "<< track.v_yaw <<std::endl;
            savefile<<track.x<<" "<<track.y<<" "<<track.z<<" "<<track.w<<" "<<track.l<<" "<<track.h<<" "<<track.yaw<<" "<<track.score<<" "<<track.class_id<<" "<<track.track_id<<std::endl;
        }
        savefile.close();

        // 保存预测轨迹
        if (savefile_future.is_open()) {
            std::vector<KF>& trackers = track_manager.get_all_trackers();
            // 先保存当前位置结果
            
            for (const auto& track : tracks) {
                // std::cout << " | Position: (" << track.x << ", " << track.y << ", " << track.z << ") \n" <<" | Dim: ("<< \
                track.w<< ", "<<track.l<<", "<<track.h<<")" <<" yaw: ("<<track.yaw<<") Class_id: ("<<track.class_id <<") class_score: ("<<track.score<<") Track_id: (" << track.track_id << ") v_yaw: "<< track.v_yaw <<std::endl;
                savefile_future<<track.x<<" "<<track.y<<" "<<track.z<<" "<<track.w<<" "<<track.l<<" "<<track.h<<" "<<track.yaw<<" "<<track.score<<" "<<track.class_id<<" "<<track.track_id<<std::endl;
            }         

            // 再保存预测结果
            for (auto& tracker : trackers) {
                const std::vector<Box3D>& future_predictions = tracker.predict_future(20);
                for (const auto& box : future_predictions) {
                        savefile_future << box.x << " " << box.y << " " << box.z <<" "
                                << box.w << " " << box.l << " " << box.h << " "
                                << box.yaw << " "
                                << box.score << " "
                                << box.class_id <<" "
                                << box.track_id << std::endl;
                }                
            }
            savefile_future.close();
        }

        } else {
            std::cerr << "Failed to open file: " << filePath << std::endl;
        }

        n++;
        // if(n >10){break;}
    }


    std::vector<Box3D> reliable_tracks = track_manager.get_reliable_tracks();
    std::vector<KF>& trackers = track_manager.get_all_trackers();

    // 文件处理结束后，统一打印所有的tracker
    // for (const auto& tracker : trackers) {
    //     // Eigen::VectorXd velocity = tracker.get_velocity();
    //     double yaw_speed = tracker.get_yaw_speed();
    //     std::cout << "Tracker ID: " << tracker.track_id << " - Velocity: " 
    //             << yaw_speed << std::endl; // 假设有一个 get_id() 方法获取 tracker 的 ID
    // }

    // 打印所有track的历史轨迹

    // std::cout << "Track history after processing all files:" << std::endl;
    // for (auto& tracker : trackers) {
    //     const std::vector<Box3D>& history = tracker.get_history();
    //     const std::vector<Box3D>& future_predictions = tracker.predict_future(10);

    //     std::cout << "Tracker ID: " << tracker.track_id << std::endl;
    //     for (const auto& box : future_predictions) {
    //         std::cout << "Position: (" << box.x << ", " << box.y << ", " << box.z << "), "
    //                 << "Dimensions: (w=" << box.w << ", l=" << box.l << ", h=" << box.h << "), "
    //                 << "class_id:(" << box.class_id << " )"
    //                 << "Yaw: " << box.yaw << ", "
    //                 << "Score: " << box.score <<", "
    //                 << "speed: " << box.v_yaw << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}