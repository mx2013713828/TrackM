# author     : mayufeng
# create time: 2024-05-21
# description: 计算3D IoU的各函数实现
# reference:   GIOU、IOU、AB3DMOT
# -[x] convex area calculation (simplify but fast version)
# -[x] 2d intersection calculation(input:box_bottom for 4 bottom points;output:intersection area)
# -[x] get corners of 3d box (inpuit:bbox(x,y,z,l,w,h,ry);output: 8 * 3(x,y,z) for 8 corners )
# -[ ] todo : example test    

import numpy as np

def box2corners(bbox):
'''
            Takes an object's 3D box with the representation of [x,y,z,l,w,h,yaw] or [x,y,z,l,w,h,yaw,score] and 
            convert it to the 8 corners of the 3D box, the box is in the lidar coordinate
            with up z.
            Returns:
                corners_3d: (8,3) array in in lidar coord

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

    l = bbox[3]
    w = bbox[4]
    z = bbox[5]
    # 先假设以坐标系原点为中心
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2] # 顺时针旋转
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    # z_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [z/2,z/2,z/2,z/2,-z/2,-z/2,-z/2,-z/2] 

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))  # (3 * 3) X (3 * 8) = 3 * 8
    corners_3d = np.transpose(corners_3d)  # 8 * 3

    return corners_3d

def convex_area(boxa_bottom, boxa_bottom):
    
	# 测试最小闭合空间，简单使用最小最大顶点
	print("use min max corners")
	xc1 = min(np.min(boxa_bottom[:, 0]), np.min(boxb_bottom[:, 0]))
	yc1 = min(np.min(boxa_bottom[:, 1]), np.min(boxb_bottom[:, 1]))
	xc2 = max(np.max(boxa_bottom[:, 0]), np.max(boxb_bottom[:, 0]))
	yc2 = max(np.max(boxa_bottom[:, 1]), np.max(boxb_bottom[:, 1]))

	convex_area = (xc2 - xc1) * (yc2 - yc1)

    return convex_area


def compute_height(corners1, corners2, inter=True):
	
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
        计算两个矩形 形成的交集. 输入排序的顶点坐标列表.
        输入坐标应按顺时针或逆时针排序
        输出坐标为逆时针排序
    """
    def inside(p, edge_start, edge_end):
        return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) > (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])

    def compute_intersection(p1, p2, p3, p4):
        denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        if denom == 0:
            return None
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
    计算两个3D检测框的 IoU。
    假设物体底面是平行的。
    参数：
        boxa_bottom, boxa_bottom: 形状为(4, 2)的数组，表示两个多边形的顶点坐标。
        
    返回值：
        iou: 多边形的 IoU。
    """    
    corners_a = box2corners(boxa_3d)            # 返回8个顶点坐标(顺时针)
    corners_b = box2corners(boxb_3d)            # 
    
	boxa_bot = corners_a[-5::-1, [0, 2]] 		# 4 x 2 底面四角坐标(逆时针)
	boxb_bot = corners_b[-5::-1, [0, 2]] 		# 4 x 2 底面四角坐标

    intersection_2d = sutherland_hodgman_clip(boxa_bot, boxb_bot)
    

    I_2D = polygon_area(intersection_2d)        ## 计算2D交集区域面积 
    C_2d = convex_area(boxa_bot, boxb_bot)      ## 计算bev最小闭合区间
    
    h_overlap = compute_height(corners_a, corners_b)                ## 计算重叠高度
    h_union   = compute_height(corners_a, corners_b, inter=False)   ## 计算联合高度

    I_3D = I_2D * h_overlap                     ## 假设物体与地面平行、物体与物体之间平行
    C_3D = C_2d * h_union                       ## 计算3D最小闭合区间

    U_3D = boxa_3d[3] * boxa_3d[4] * boxa_3d[5] + boxb_3d[3] * boxb_3d[4] * boxb_3d[5] - I_3D ## BOXa体积 + BOXb体积 - I_3D

    GIOU = I_3D / U_3D - (C_3D - U_3D) / C_3D

    return GIOU




