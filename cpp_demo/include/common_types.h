/*
 * File:        common_types.h
 * Author:      Yufeng Ma
 * Date:        2026-01-09
 * Email:       97357473@qq.com
 * Description: Common data structures and types used across the tracking project.
 */

#pragma once

#include <vector>
#include <array>
#include <Eigen/Dense>

#include "../cuda_pointpillar/postprocess.h"
#include "../cuda_centerpoint/postprocess.h"
#include "../lidar_utility.h"

enum class LIDAR_DET_TYPE { // define see params.h    
    CONE     = 0,   // size:[0.5, 0.8]  	// 锥桶 
    PEOPLE   = 1,   // size:[0.71, 0.8]  	// 行人 
    CAR_S    = 2,   // size:[4.1, 1.8]   	// 社会车 
    CAR_P    = 3,   // size:[5.3, 2.1]  	// 皮卡  
    TRUCK    = 4,   // size:[9.6, 4.3]  	// 矿卡
    TRUCK_S  = 5,   // size:[7.3, 2.5]      // 小型卡车 
    TRUCK_M  = 6,   // size:[12.2, 3.3]  	// 中型卡车
    TRUCK_L  = 7,   // size:[17.9, 4.1]  	// 半挂车
    TRUCK_F  = 8,   // size:[8.0, 3.0]   	// 油罐车，洒水车
    EXCAVATOR= 9,   // size:[11.6, 4.0]	    // 挖掘机    
    LOADER   = 10,   // size:[9.4, 3.4]   	// 装载机
    ROLLER   = 11,  // size:[8.0, 3.0]  	// 压路机
    GRADER   = 12,  // 平地机
};

typedef struct 
{
    float x;
    float y;
    float z;    //　坐标点会用到该数据结构　　雷达点会用到该数据结构,　根据实际情况使用不同字段
    float l; 
    int   conf; // 置信度
} point_t;

struct target_t {
    int    classid;

    int    x_pixel;    
    int    y_pixel;
    int    w_pixel;
    int    h_pixel;
    float  conf;

    float  x_world;    // 车辆坐标系 左右偏移距离（单位：米)
    float  y_world;    // 车辆坐标系 前后距离（单位：米)
    float  z_world;    // 车辆坐标系 中心点高度(单位：米)
    float  w_world;    // 宽度  若雷达障碍物, 则为矩形框的宽
    float  l_world;    // 长度  若雷达障碍物, 则为矩形框的长
    float  h_world;    // 高度　若雷达障碍物, 则为矩形框的高度

    float  x_world1;   // horizontal plane 
    float  y_world1;
    float  w_world1;
    float  h_world1;
    float  l_world1;

    float  x_world2;   // triangle
    float  y_world2;
    float  w_world2;
    float  h_world2;
    float  l_world2;
    
    float  x_earth;    // 绝对坐标系或大地坐标系 若雷达障碍物, 则为矩形框中心点位置的坐标
    float  y_earth;    // 绝对坐标系或大地坐标系 若雷达障碍物, 则为矩形框中心点位置的坐标
    float  z_earth;    // 绝对坐标系或大地坐标系,若雷达障碍物, 则为矩形框中心点位置的坐标
    float  heading_world;    // 航向弧度  在激光雷达障碍物使用此数据段时仅表示车辆坐标系下的航向，不代表大地坐标系下的航向
    float  heading_earth; // 航向弧度  绝对坐标系或大地坐标系 若雷达障碍物, 则为障碍物在大地坐标系下的绝对航向 [-180 +180] 正北: 0度 正东:90度 正西:-90度
    std::vector<point_t> points_earth; // 障碍物矩形框四个顶点的大地坐标系的值
    std::vector<point_t> points_world; // 障碍物矩形框四个顶点的车辆坐标系的值
    float  vx;         // km/h
    float  vy;         // km/h
    float  speed;      // km/h
    int    property;   // 运动属性 比如 同向 交叉 等
    float  k;          // k = x / y
    float  s;          // 1:left 2:right 0:undefined
    int    frames;     // 连续识别到几帧
    int    track_id;
    int    time_since_update; // 最近更新时间

    std::vector<point_t> points_world_predict; // 预测的该障碍物车辆坐标系下的轨迹
    std::vector<point_t> points_earth_predict; // 预测的该障碍物大地坐标系下的轨迹

    long   time_stamp;

    // 默认构造函数进行零初始化
    target_t() : classid(0), x_pixel(0), y_pixel(0), w_pixel(0), h_pixel(0), conf(0.0f),
                 x_world(0.0f), y_world(0.0f), z_world(0.0f), w_world(0.0f), l_world(0.0f), h_world(0.0f),
                 x_world1(0.0f), y_world1(0.0f), w_world1(0.0f), h_world1(0.0f), l_world1(0.0f),
                 x_world2(0.0f), y_world2(0.0f), w_world2(0.0f), h_world2(0.0f), l_world2(0.0f),
                 x_earth(0.0f), y_earth(0.0f), z_earth(0.0f), heading_world(0.0f), heading_earth(0.0f),
                 vx(0.0f), vy(0.0f), speed(0.0f), property(0), k(0.0f), s(0.0f), frames(0), track_id(-1),
                 time_stamp(0) {}
};


struct Bndbox {
    float x;    // 中心点坐标
    float y;    // 中心点坐标
    float z;    // 中心点坐标
    float w;
    float l;
    float h;
    float rt;   // 航向 单位：弧度【-pi -- +pi】
    int id;
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};

struct Box3D {
    float x, y, z, w, l, h, vx,vy,yaw;
    int class_id;
    float score;
    int track_id;
    float v_yaw;
    // 默认构造函数
    Box3D() : x(0), y(0), z(0), w(0), l(0), h(0), yaw(0), class_id(-1), score(0), track_id(-1), v_yaw(0) {}

    // 从 Eigen::VectorXd 初始化 Box3D
    Box3D(const Eigen::VectorXd& bbox, int class_id_ = -1, float score_ = 0, int track_id_ = -1, float v_yaw_ = 0) {
        x = bbox(0);
        y = bbox(1);
        z = bbox(2);
        w = bbox(3);
        l = bbox(4);
        h = bbox(5);
        yaw = bbox(6);
        class_id = class_id_;
        score = score_;
        track_id = track_id_;
        v_yaw = v_yaw_;
    }

    // 从 Bndbox 初始化 Box3D beishan使用的检测结果结构
    Box3D(const Bndbox& bndbox) {
        x = bndbox.x;
        y = bndbox.y;
        z = bndbox.z;
        w = bndbox.w;
        l = bndbox.l;
        h = bndbox.h;
        yaw = bndbox.rt;
        class_id = bndbox.id;
        score = bndbox.score;
        track_id = -1;
    }

    // 拷贝构造函数
    Box3D(const Box3D& other) = default;
    Box3D(Box3D&& other) = default;
    Box3D& operator=(const Box3D& other) = default;
    Box3D& operator=(Box3D&& other) = default;

    // 添加新的构造函数
    Box3D(float x_, float y_, float z_, float w_, float l_, float h_, float yaw_,
          int class_id_ = -1, float score_ = 0, int track_id_ = -1, float v_yaw_ = 0)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), yaw(yaw_),
          class_id(class_id_), score(score_), track_id(track_id_), v_yaw(v_yaw_) {}
};
