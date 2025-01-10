## TrackM 整体结构组织

### UML 类图

```
+-------------------------------------+
|             Filter                  |
+-------------------------------------+
| - initial_pos                       |
| - time_since_update                 |
| - id (tracker_id)                   |
| - hits                              |
| - info dict{class_id:int,score:float}|
+-------------------------------------+
| + Filter(bbox_3d, info, ID)         |
+-------------------------------------+

     ^ (inherits)
     |
+-------------------------------------+
|                 KF                  |
+-------------------------------------+
| - kf                                |
+-------------------------------------+
| + KF(bbox_3d, info, ID)             |
| + _init_kalman_filter()             |
| + update()                          |
| + predict()                         |
| + get_state()                       |
+-------------------------------------+

     ^ (manages *)
     |
+-------------------------------------+
|          TrackerManager             |
+-------------------------------------+
| - trackers                          |
| - next_id                           |
| - max_age                           |
| - min_hits                          |
+-------------------------------------+
| + TrackerManager(max_age, min_hits) |
| + update(detections)                |
| + increment_age_unmatched_trackers()|
| + create_new_trackers()             |
| + get_tracks()                      |
| + update_trackers()                 |
+-------------------------------------+

```

### 主要函数

- associate_detections_to_trackers
   - 1.使用GIOU计算“跟踪预测的结果”与“检测结果”的IOU,多目标下会形成一个2维矩阵。
   - 2.使用hungarian algorithm（匈牙利匹配）,对“预测结果”与“匹配结果”进行匹配。
   - 3.对每个匹配对,根据GIOU的值，将其划分到matches,unmatched_dets,unmatched_trks

- sutherland_hodgman_clip : 计算两个矩形形成的交集。输入排序的顶点坐标列表

- polygon_area : 计算多边形面积,使用高斯面积公式

- compute_height : 获取两个框的联合高度或者交集高度

- convex_area : 计算两个矩形的最小闭合空间的面积

- box2corners : 将物体的3D边界框转换为3D边界框的8个角点,输入:box_3d[x, y, z, l, w, h, yaw],输出:8个角点(8, 3)

- calculate_iou : 获取两个box_3d的GIOU,输入(boxa_3d,boxb_3d),使用其他函数计算GIOU

### 主要类

- Filter 滤波器基类 
   - 成员函数
      - 1 构造函数 : 使用参数{bbox_3d, info, ID}初始化类内成员变量
   - 成员变量
      - initial_pos
      - time_since_update
      - id (tracker_id)
      - hits
      - info dict{'class_id':int,'score':float}

- KF 卡尔曼滤波器类(继承Filter类)
   - 成员变量
      - kf(卡尔曼滤波器对象)
   - 成员函数
      - 构造函数 : 调用继承父类的构造函数{bbox_3d, info, ID}、调用_init_kalman_filter、初始化kf.x[:7] = initial_pos.reshape((7, 1))
      - _init_kalman_filter ：初始化卡尔曼滤波器对象
      - update : 更新kf的状态
      - predict : 预测kf下一步 (kf.predict)
      - get_state : 获取当前kf的X

- TrackerManager 跟踪管理类
   - 成员变量
      - trackers : 包含目前所有跟踪器的列表
      - next_id : 下一次创建跟踪器的id
      - max_age : 未被匹配跟踪器的最大存活时间
      - min_hits : 跟踪器匹配的最小次数

   - 成员函数
      - 构造函数 : 参数为max_age、min_hits.初始化成员变量:trackers=[],next_id=1,max_age = max_age,min_hits = min_hits
      - update : 参数为detections,原地更新trackers以及其中每个tracker的状态.
         1. 若trackers为空,使用detections初始化trackers,否则执行后续步骤
         2. 使用trackers.predict()预测轨迹
         3. 使用associate_detections_to_trackers 进行数据关联匹配,获取matches、unmatched_detections、unmatched_trackers
         4. 对命中的跟踪器更新kf状态
         5. 对未命中的检测框创建新的跟踪器
         6. 增加未命中tracker的age
         7. 最后,删除age超过max_age的跟踪器

      - increment_age_unmatched_trackers : 增加未命中tracker的age
      - create_new_trackers : 创建新的跟踪器for unmatched detections
      - get_tracks : 
      - get_all_trackers : 返回所有跟踪器
      - update_trackers : 对命中的跟踪器更新kf状态

