import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

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

# 读取点云时设置范围
def load_point_cloud_within_range(file_path, x_range=None, y_range=None, z_range=None):
    pc = parse_pcd(file_path)  # 假设这是你现有的读取点云的函数
    
    if x_range is not None:
        pc = pc[(pc[:, 0] >= x_range[0]) & (pc[:, 0] <= x_range[1])]
    if y_range is not None:
        pc = pc[(pc[:, 1] >= y_range[0]) & (pc[:, 1] <= y_range[1])]
    if z_range is not None:
        pc = pc[(pc[:, 2] >= z_range[0]) & (pc[:, 2] <= z_range[1])]
    
    return pc

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

# 解析未来轨迹的函数
def parse_future_predictions(prediction_file):
    future_predictions = []
    with open(prediction_file, 'r') as f:
        lines = f.readlines()
        track_predictions = []
        for line in lines:
            # 解析预测框
            values = list(map(float, line.strip().split()))
            future_predictions.append({
                'x': values[0], 'y': values[1], 'z': values[2],
                'l': values[3], 'w': values[4], 'h': values[5],
                'yaw': values[6], 'score': values[7], 'class_id': int(values[8]), 
                'track_id': int(values[9])
            })
    return future_predictions


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

# 4. 自动播放点云和检测框并保存为视频
def visualize_with_video_output(pcd_folder, detection_files, video_path, prediction_folder=None,pause_time=0.1, 
                                x_range=None, y_range=None, z_range=None):
    fig = plt.figure(figsize=(24, 24))  # 增大显示窗口
    writer = FFMpegWriter(fps=int(1/pause_time))  # 使用 FFmpeg 保存视频

    with writer.saving(fig, video_path, dpi=100):
        # for pcd_file, detection_file in zip(pc_files, detection_files):
        for detection_file in detection_files:
            plt.clf()  # 清除之前的图像以显示新帧
            pcd_file = os.path.join(pcd_folder, os.path.basename(detection_file).split('.')[0] + '.pcd')
            print(f"processing point cloud : {os.path.basename(pcd_file)}")
            # 加载并过滤点云
            pc = load_point_cloud_within_range(pcd_file, x_range=x_range, y_range=y_range, z_range=z_range)

            # 加载检测框
            detections = parse_detection(detection_file)
            
            # 读取对应时刻的未来轨迹
            if prediction_folder != None:
                prediction_file = os.path.join(prediction_folder, os.path.basename(detection_file).replace('cpp_result', 'cpp_result_future'))
                future_predictions = parse_future_predictions(prediction_file)

            # 绘制点云 (x, y)，增大点的大小
            plt.scatter(pc[:, 0], pc[:, 1], s=0.1, c='gray', label='Point Cloud')

            # 绘制检测框
            for detection in detections:
                corners = box_to_corners_2d(detection)  
                
                # 转换为 NumPy 数组，以便进行切片
                corners = np.array(corners)
                corners = np.concatenate([corners, corners[0:1, :]])  # 闭合四边形

                # 绘制边界框
                plt.plot(corners[:, 0], corners[:, 1], color='red')

                # 计算车头箭头的起始点和终止点
                cx, cy = detection['x'], detection['y']
                yaw = detection['yaw']
                arrow_length = 4.0  # 增加箭头长度
                arrow_dx = arrow_length * np.cos(yaw)
                arrow_dy = arrow_length * np.sin(yaw)

                # 绘制车头箭头
                plt.arrow(cx, cy, arrow_dx, arrow_dy, 
                          head_width=0.7, head_length=0.8, fc='orange', ec='orange')

                # 在框上显示 class_id 和 track_id
                plt.text(cx, cy + 1, f"CID: {detection['class_id']}, TID: {detection['track_id']}",
                         fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.5))

            # 绘制未来轨迹的中心点
            if prediction_folder:
                for future_track in future_predictions:
                    # 获取每个预测框的中心点坐标
                    center_x = future_track['x']
                    center_y = future_track['y']
                    
                    # 绘制中心点，设置颜色、大小和透明度等
                    plt.scatter(center_x, center_y, color='blue', marker='o', alpha=0.5)


            # 在左上角显示当前帧文件名
            plt.text(0.01, 0.99, f"Frame: {os.path.basename(pcd_file)}", 
                     fontsize=12, color='black', transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))


            plt.axis('equal')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('2D Bird\'s Eye View (BEV)')
            plt.grid(True)

            writer.grab_frame()  # 记录当前帧

# 5. 处理文件夹中的 PCD 和检测文件
def process_folders_with_video_output(pcd_folder, detection_folder, video_path, prediction_folder = None,pause_time=0.1, 
                                      x_range=None, y_range=None, z_range=None):
    # 获取所有文件
    # pcd_files = sorted([os.path.join(pcd_folder, f) for f in os.listdir(pcd_folder) if f.endswith('.pcd') or f.endswith('.bin')])
    detection_files = sorted([os.path.join(detection_folder, f) for f in os.listdir(detection_folder) if f.endswith('.txt')])

    # 可视化和保存为视频
    visualize_with_video_output(pcd_folder, detection_files, video_path, prediction_folder = prediction_folder, pause_time=pause_time, 
                                x_range=x_range, y_range=y_range, z_range=z_range)


import argparse

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Visualize PCD and detection results.')
    parser.add_argument('--pcd_folder', type=str, required=True,
                        help='Path to the folder containing PCD files.')
    parser.add_argument('--detection_folder', type=str, required=True,
                        help='Path to the folder containing detection result TXT files.')
    parser.add_argument('--prediction_folder', type=str, default=None,
                        help='Path to the folder containing PCD files.')
    parser.add_argument('--video_path', type=str, required=True, help="保存视频的路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用处理文件夹的函数
    # process_folders(args.pcd_folder, args.detection_folder)
    
    process_folders_with_video_output(args.pcd_folder, args.detection_folder, args.video_path, args.prediction_folder,
                                      pause_time=0.25, 
                                      x_range=[-42,42], y_range=[-30,78], z_range=[-2,8])