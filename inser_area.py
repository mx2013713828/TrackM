import numpy as np

def polygon_area(vertices):
    """Calculate the area of a polygon given its vertices using the Shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """Clip a polygon with another polygon using the Sutherland-Hodgman algorithm."""
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