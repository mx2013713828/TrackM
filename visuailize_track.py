import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# 1. 解析点云文件（PCD 或 BIN 格式）
def parse_pcd(pcd_file):
    _, ext = os.path.splitext(pcd_file)  # 获取文件扩展名
    if ext.lower() == '.pcd':
        pcd = o3d.io.read_point_cloud(pcd_file)
        return np.asarray(pcd.points)
    elif ext.lower() == '.bin':
        # 假设每个点有3个坐标（x, y, z），可以根据具体的格式调整
        pc = np.fromfile(pcd_file, dtype=np.float32)
        return pc.reshape(-1, 4)  # 假设点的格式为 (x, y, z)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# 2. 解析检测框文件（TXT）
def parse_detection(txt_file):
    detections = []
    with open(txt_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            detection = {
                'x': values[0], 'y': values[1], 'z': values[2],  # 位置
                'l': values[3], 'w': values[4], 'h': values[5],  # 宽, 长, 高
                'yaw': (values[6]),  # 偏航角
                'score': values[7],  # 检测得分
                'class_id': int(values[8]),  # 类别 ID
                'track_id': int(values[9])  # 跟踪 ID
            }
            detections.append(detection)
    return detections

# 3. 转换检测框到2D角点
def box_to_corners_2d(bbox):
    """ 计算底部四个角点的坐标
    """
    bottom_center = np.array([bbox['x'], bbox['y'], bbox['z'] - bbox['h'] / 2])
    cos, sin = np.cos(bbox['yaw']), np.sin(bbox['yaw'])
    
    # 计算四个角点
    pc0 = np.array([bbox['x'] + cos * bbox['l'] / 2 + sin * bbox['w'] / 2,
                    bbox['y'] + sin * bbox['l'] / 2 - cos * bbox['w'] / 2,
                    bbox['z'] - bbox['h'] / 2])
    
    pc1 = np.array([bbox['x'] + cos * bbox['l'] / 2 - sin * bbox['w'] / 2,
                    bbox['y'] + sin * bbox['l'] / 2 + cos * bbox['w'] / 2,
                    bbox['z'] - bbox['h'] / 2])
    
    pc2 = 2 * bottom_center - pc0
    pc3 = 2 * bottom_center - pc1

    return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]

# 4. 可视化点云和检测框
def visualize(pc, detections, pause_time=0.25):
    plt.figure(1,figsize=(36, 36))  # 增大显示窗口

    # 清除之前的图像，以显示新帧
    plt.clf()     
    # 绘制点云 (x, y)，增大点的大小
    plt.scatter(pc[:, 0], pc[:, 1], s=0.01, c='gray', label='Point Cloud')  # 增大s参数

    # 绘制检测框
    for detection in detections:
        corners = box_to_corners_2d(detection)  # 使用新的方法计算角点
        
        # 转换为 NumPy 数组，以便进行切片
        corners = np.array(corners)
        corners = np.concatenate([corners, corners[0:1, :]])  # 闭合四边形

        # 绘制边界框
        plt.plot(corners[:, 0], corners[:, 1], color='red')

        # 计算车头箭头的起始点和终止点
        cx, cy = detection['x'], detection['y']
        yaw = detection['yaw']
        arrow_length = 3.0  # 增加箭头长度
        arrow_dx = arrow_length * np.cos(yaw)  # 箭头在 x 方向的变化量
        arrow_dy = arrow_length * np.sin(yaw)  # 箭头在 y 方向的变化量
        
        # 绘制车头箭头
        plt.arrow(cx, cy, arrow_dx, arrow_dy, 
                  head_width=0.9,  # 增加箭头头部的宽度
                  head_length=0.9,  # 增加箭头头部的长度
                  fc='orange', ec='orange')  # 使用更亮的颜色
        # 在框上显示 class_id 和 track_id
        # plt.text(cx, cy, f"ID: {detection['class_id']}, Track: {detection['track_id']}",
        #          fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('equal')
    plt.grid(True)  # 添加网格以增强可视化效果

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Bird\'s Eye View (BEV)')
    # plt.show(block=False) 
    plt.pause(pause_time)  # 设置自动播放每帧的间隔时间

# 5. 处理文件夹中的 PCD 和检测文件
def process_folders(pcd_folder, detection_folder):
    pcd_files = sorted([f for f in os.listdir(pcd_folder)])
    detection_files = sorted([f for f in os.listdir(detection_folder) if f.endswith('.txt')])

    for pcd_file, detection_file in zip(pcd_files, detection_files):
        pcd_path = os.path.join(pcd_folder, pcd_file)
        detection_path = os.path.join(detection_folder, detection_file)
        print(pcd_path)

        # 解析点云和检测框
        pc = parse_pcd(pcd_path)
        detections = parse_detection(detection_path)

        # 可视化
        print(f"Visualizing {pcd_file} and {detection_file}")
        visualize(pc, detections)

import argparse

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Visualize PCD and detection results.')
    parser.add_argument('--pcd_folder', type=str, required=True,
                        help='Path to the folder containing PCD files.')
    parser.add_argument('--detection_folder', type=str, required=True,
                        help='Path to the folder containing detection result TXT files.')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用处理文件夹的函数
    process_folders(args.pcd_folder, args.detection_folder)

