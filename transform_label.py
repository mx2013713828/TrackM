from calibration_kitti import Calibration
from itertools import groupby
import numpy as np
def get_calib(calib_file):
    # 读取一个视频序列的标定文件
    return Calibration(calib_file)

def group_by_first_element(arr):
    # 先对数组按每行的第一个元素进行排序
    arr_sorted = arr[arr[:, 0].argsort()]
    
    # 然后使用 groupby 进行分组
    grouped = {}
    for key, group in groupby(arr_sorted, lambda x: x[0]):
        grouped[key] = np.array(list(group))
    
    return grouped

def get_track_label(label_file):
    # 读取一个视频序列的标签,按frame_id排序,且转换loc到雷达坐标系,并分开保存到000001.txt,000002.txt...
    class_to_id = {'Pedestrian':0, 'Car':1, 'Van':2}

    calib_file = '/home/sdlg/mayufeng/fast_myf/TrackData/kitti/data_tracking_calib/training/calib/0000.txt'
    calib = get_calib(calib_file)

    with open('data/0000.txt','r') as f:
        lines = f.readlines()
        all_labels = list()
        for line in lines:
            label_frame = line.split()
            all_labels.append(label_frame)
        grouped = group_by_first_element(np.array(all_labels))
        
        for key,group in grouped.items():
            frame_id = ('000000' + key )[-6:]
            with open('data/labels/0000/'+frame_id+'.txt','w') as fw:
                for line in group:
                    class_name = line[2]
                    if class_name in class_to_id.keys():
                        class_id = class_to_id[class_name]
                        loc = np.array([[float(line[13]), float(line[14]), float(line[15])]])

                        h = float(line[10])
                        w = float(line[11])
                        l = float(line[12])
                        print('loc:' , loc[0])
                        loc_lidar = calib.rect_to_lidar(loc)
                        print('loc_lidar:' , loc_lidar[0])
                        loc_lidar[:, 2] += h/2 # 调整中心点高度从底部到中心
                        yaw = - (np.pi/2 + float(line[16])) # 调整yaw方向

                        fw.write(str(loc_lidar[0][0]) + ' ' + str(loc_lidar[0][1]) + ' ' + str(loc_lidar[0][2]) + ' ' + str(l) + ' ' + str(w) + ' ' + str(h) + ' ' + str(yaw) + ' ' + str(class_id) + ' ' + str(1) + '\n')


if __name__ == '__main__':
    print("transform lidar from cam to lidar")
    get_track_label('./data/0000.txt')
