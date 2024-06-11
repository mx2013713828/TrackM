# <div align='center'> ⭐TrackM⭐ </div>
  
<div align = "center"> <img src="https://pic.imgdb.cn/item/65dc5dfc9f345e8d03446103.png" height=100 width=300> </div>

#### <p align = "center">![Static Badge](https://img.shields.io/badge/mayufeng-blue?style=flat&label=Author)![Static Badge](https://img.shields.io/badge/2024/06/01-blue?style=flat&label=CreateTime)![Static Badge](https://img.shields.io/badge/97357473@qq\.com\-blue?style=flat&label=Email)</p>

这是多目标跟踪器TrackM的代码实现, 算法思路参考了[deep_sort](https://github.com/nwojke/deep_sort)，[AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)。增加了C++版本，可直接将检测结果作为输入。

---


## TODO

- [ ] python 实现
   - [x] GIOU
   - [x] convex 最小最大点实现
   - [x] convex 凸包实现(借助第三方库)
   - [x] TrackerManager 跟踪器生命周期管理类
   - [ ] 返回track结果包含类别名(Filter类中加class_id参数)
   - [ ] 进行匹配时考虑检测框的类别
   - [ ] 保存txt 分隔符从','改为空格;
   - [ ] 实时可视化跟踪
- [ ] C++ 实现
   - [x] C++ giou代码
   - [x] convex 最小最大点实现
   - [x] C++ 卡尔曼滤波器类
   - [x] C++ KF 跟踪器类
   - [x] C++ TrackerManager 跟踪器生命周期管理类
   - [ ] 点云流检测结果测试保存结果到txt
   - [ ] 可视化跟踪
   - [ ] 动态链接库



##  附录

### kitti track format

```
#Values    Name      Description
----------------------------------------------------------------------------
   1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries.
                     Truncation 2 indicates an ignored object (in particular
                     in the beginning or end of a track) introduced by manual
                     labeling.
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
```

