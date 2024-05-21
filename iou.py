import numpy as np

def box2corners(bbox):
'''
            Takes an object's 3D box with the representation of [x,y,z,l,w,h,yaw] and 
            convert it to the 8 corners of the 3D box, the box is in the lidar coordinate
            with up z.
            Returns:
                corners_3d: (8,3) array in in rect camera coord

            box corner order is like follows
                    1 -------- 0         top is top because z direction is positive
                   /|         /|
                  2 -------- 3 .
                  | |        | |
                  . 5 -------- 4
                  |/         |/
                  6 -------- 7    
            
'''    


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
		height = max(0.0, ymax - ymin)
	else:			# compute union height
		zmax = max(corners1[0, 2], corners2[0, 2])
		zmin = min(corners1[4, 2], corners2[4, 2])
		height = max(0.0, ymax - ymin)

	return height

def calculate_iou(boxa_3d, boxb_3d):
    """
    计算两个多边形的 IoU。
    
    参数：
        boxa_bottom, boxa_bottom: 形状为(4, 2)的数组，表示两个多边形的顶点坐标。
        
    返回值：
        iou: 多边形的 IoU。
    """    
    corners_a = box2corners(boxa_3d)
    corners_b = box2corners(boxb_3d)
    
	boxa_bot = corners_a[-5::-1, [0, 2]] 		# 4 x 2
	boxb_bot = corners_b[-5::-1, [0, 2]] 		# 4 x 2

    I_2D = compute_inter_2D(boxa_bot, boxb_bot) ## 计算2D交集 待实现
    C_2d = convex_area(boxa_bot, boxb_bot)      ## 计算bev最小闭合区间
    
    h_overlap = compute_height(corners_a, corners_b)                ## 计算重叠高度
    h_union   = compute_height(corners_a, corners_b, inter=False)   ## 计算联合高度

    I_3D = I_2D * h_overlap 
    C_3D = C_2d * h_union                       ## 计算3D最小闭合区间

    U_3D = boxa_3d[3] * boxa_3d[4] * boxa_3d[5] + boxb_3d[3] * boxb_3d[4] * boxb_3d[5] - I_3D ## BOXa体积 + BOXb体积 - I_3D

    GIOU = I_3D / U_3D - (C_3D - U_3D) / C_3D

    return GIOU




