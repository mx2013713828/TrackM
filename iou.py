# author     : mayufeng
# create time: 2024-05-21
# description: 计算3D IoU的各函数实现
# reference:   GIOU、IOU、AB3DMOT
# -[x] convex area calculation (simplify but fast version)
# -[x] 2d intersection calculation(input:box_bottom for 4 bottom points;output:intersection area)
# -[x] get corners of 3d box (inpuit:bbox(x,y,z,l,w,h,ry);output: 8 * 3(x,y,z) for 8 corners )
# -[x] example test    

import numpy as np
from scipy.spatial import ConvexHull # 凸包 测试两种计算闭合空间的方法

def box2corners(bbox):
    '''
    将物体的3D边界框从 [x, y, z, l, w, h, yaw] 或 [x, y, z, l, w, h, yaw, score] 格式
    转换为3D边界框的8个角点。边界框在 LiDAR 坐标系中,z 轴向上。

    参数:
        bbox (数组或列表): 边界框参数 [x, y, z, l, w, h, yaw] 或 [x, y, z, l, w, h, yaw, score]

    返回:
        np.ndarray: (8, 3) 数组，表示 3D 边界框在 LiDAR 坐标系中的 8 个角点

        box corner order is like follows
            1 -------- 0         top is top because z direction is positive
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7    
            
    x->l,y->w,z->h
            
    '''    

    yaw = bbox[6]
    c = np.cos(yaw)
    s = np.sin(yaw)

    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    x, y, z = bbox[0], bbox[1], bbox[2]
    l, w, h = bbox[3], bbox[4], bbox[5]

    # 先假设以坐标系原点为中心
    # 3d bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2] # 顺时针旋转
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    # z_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))  # (3 * 3) X (3 * 8) = 3 * 8
    corners_3d[0,:] = corners_3d[0,:] + x
    corners_3d[1,:] = corners_3d[1,:] + y
    corners_3d[2,:] = corners_3d[2,:] + z
    corners_3d = np.transpose(corners_3d)  # 8 * 3

    return corners_3d

def convex_area(boxa_bottom, boxb_bottom):  
    """  
    计算最小闭合空间的面积
    ## 相比较使用凸包算法,速度更快,但是精度略低,最终的giou值也比正确的giou要低,因此giou_thres也要设置第一点
    """
    xc1 = min(np.min(boxa_bottom[:, 0]), np.min(boxb_bottom[:, 0]))
    yc1 = min(np.min(boxa_bottom[:, 1]), np.min(boxb_bottom[:, 1]))
    xc2 = max(np.max(boxa_bottom[:, 0]), np.max(boxb_bottom[:, 0]))
    yc2 = max(np.max(boxa_bottom[:, 1]), np.max(boxb_bottom[:, 1]))
    convex_area = (xc2 - xc1) * (yc2 - yc1)

    # # print("use convexhull ")
    # all_corners = np.vstack((boxa_bottom, boxb_bottom))
    # C = ConvexHull(all_corners)
    # convex_corners = all_corners[C.vertices]

    # convex_area = polygon_area(convex_corners)

    return convex_area


def compute_height(corners1, corners2, inter=True):
	"""
    获取联合高度或者交集高度
    """
	if inter: 		# compute overlap height
		zmax = min(corners1[0, 2], corners2[0, 2])
		zmin = max(corners1[4, 2], corners2[4, 2])
		height = max(0.0, zmax - zmin)
	else:			# compute union height
		zmax = max(corners1[0, 2], corners2[0, 2])
		zmin = min(corners1[4, 2], corners2[4, 2])
		height = max(0.0, zmax - zmin)

	return height


def polygon_area(vertices):
    """
    计算多边形面积
    高斯面积公式
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """
    计算两个矩形形成的交集。输入排序的顶点坐标列表。
    输入坐标应按逆时针排序。
    输出坐标为逆时针排序。
    """
    def inside(p, edge_start, edge_end):
        # 检查点 p 是否在边 edge_start -> edge_end 的内部
        return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) >= (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])

    def compute_intersection(p1, p2, p3, p4):
        # 计算两条边 p1->p2 和 p3->p4 的交点
        denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        if denom == 0:
            return None  # 平行或重叠，无交点
        x = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denom
        y = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])) / denom
        return np.array([x, y])

    output_list = subject_polygon
    for i in range(len(clip_polygon)):
        input_list = output_list
        output_list = []
        if len(input_list) == 0:
            break
        edge_start = clip_polygon[i]
        edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
        for j in range(len(input_list)):
            current_point = input_list[j]
            prev_point = input_list[(j - 1) % len(input_list)]
            if inside(current_point, edge_start, edge_end):
                if not inside(prev_point, edge_start, edge_end):
                    intersection = compute_intersection(prev_point, current_point, edge_start, edge_end)
                    if intersection is not None:
                        output_list.append(intersection)
                output_list.append(current_point)
            elif inside(prev_point, edge_start, edge_end):
                intersection = compute_intersection(prev_point, current_point, edge_start, edge_end)
                if intersection is not None:
                    output_list.append(intersection)

    return np.array(output_list)

def calculate_iou(boxa_3d, boxb_3d):
    """
    计算两个3D检测框的 GIoU。
    假设物体底面是平行的。
    参数：
        boxa_bottom, boxa_bottom: 形状为(4, 2)的数组，表示两个多边形的顶点坐标。
        
    返回值：
        GIOU,IOU3d,IOU2d . giou可以在交集为0的情况下衡量两个框的距离,交集为0时,giou为负数,离得越近,giou越趋向于0,值越大。
    """    
    corners_a = box2corners(boxa_3d)            # 返回8个顶点坐标(顺时针)
    corners_b = box2corners(boxb_3d)            # 

    boxa_bot = corners_a[-5::-1, [0, 1]] 		# 4 x 2 底面四角坐标(逆时针) 重要
    boxb_bot = corners_b[-5::-1, [0, 1]] 		# 4 x 2 底面四角坐标

    intersection_2d = sutherland_hodgman_clip(boxa_bot, boxb_bot)
    if len(intersection_2d) == 0:               ## 没有交集
        I_2D = 0
    else:
        I_2D = polygon_area(intersection_2d)    ## 计算2D交集区域面积 
    C_2d = convex_area(boxa_bot, boxb_bot)      ## 计算bev最小闭合区间

    h_overlap = compute_height(corners_a, corners_b)                ## 计算重叠高度
    h_union   = compute_height(corners_a, corners_b, inter=False)   ## 计算联合高度

    I_3D = I_2D * h_overlap                     ## 假设物体与地面平行、物体与物体之间平行
    C_3D = C_2d * h_union                       ## 计算3D最小闭合区间
    
    U_2D = boxa_3d[3] * boxa_3d[4] + boxb_3d[3] * boxb_3d[4] - I_2D
    U_3D = boxa_3d[3] * boxa_3d[4] * boxa_3d[5] + boxb_3d[3] * boxb_3d[4] * boxb_3d[5] - I_3D ## BOXa体积 + BOXb体积 - I_3D
    
    # print(f"Box A Bottom Corners: \n{boxa_bot}")
    # print(f"Box B Bottom Corners: \n{boxb_bot}")

    # print(f"I_2D: {I_2D}")
    # print(f"I_3D: {I_3D}")
    # print(f"C_3D: {C_3D}")
    # print(f"U_2D: {U_2D}")
    # print(f"U_3D: {U_3D}")

    IOU2d = I_2D / U_2D
    IOU3d = I_3D / U_3D
    GIOU = I_3D / U_3D - (C_3D - U_3D) / C_3D

    return GIOU,IOU3d,IOU2d

if __name__ == '__main__':

    print("why running iou.py?")


