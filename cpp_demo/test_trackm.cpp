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
    std::string folderPath = "../data/detections/test/"; // 替换为你的文件夹路径
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
        tracks = track_manager.get_tracks();
    
        std::ofstream savefile("../data/cpp_result/" + filePath);
        for (const auto& track : tracks) {
            std::cout << " | Position: (" << track.x << ", " << track.y << ", " << track.z << ") \n" <<" | Dim: ("<< \
            track.w<< ", "<<track.l<<", "<<track.h<<")" <<" yaw: ("<<track.yaw<<") Class_id: ("<<track.class_id <<") class_score: ("<<track.score<<") Track_id: " << track.track_id << std::endl;
            savefile<<track.x<<" "<<track.y<<" "<<track.z<<" "<<track.w<<" "<<track.l<<" "<<track.h<<" "<<track.yaw<<" "<<track.class_id<<" "<<track.score<<" "<<track.track_id<<std::endl;
        }
        
        } else {
            std::cerr << "Failed to open file: " << filePath << std::endl;
        }
        n++;
        // if(n >10){break;}
    }

    return 0;
}
