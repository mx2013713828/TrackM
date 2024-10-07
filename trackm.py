import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import iou

def limit_angle(ry):
    """
    将角度 ry 限定在 [-pi, pi] 范围内
    :param ry: 输入的角度值（单位：弧度）
    :return: 限定在 [-pi, pi] 范围内的角度值
    """
    ry = np.fmod(ry, 2 * np.pi)  # 将角度值限制在 [-2*pi, 2*pi] 之间
    if ry > np.pi:
        ry -= 2 * np.pi  # 如果角度大于 pi，则减去 2*pi
    elif ry < -np.pi:
        ry += 2 * np.pi  # 如果角度小于 -pi，则加上 2*pi
    return ry

class Filter:
    def __init__(self, bbox3D: np.ndarray, info: dict, ID: int):
        self.initial_pos = bbox3D 
        self.time_since_update = 0 
        self.id = ID 
        self.hits = 1  # Number of total hits including the first detection 
        self.info = info  # Other associated information 
        
class KF(Filter):
    def __init__(self, bbox3D: np.ndarray, info: dict, ID: int):
        super().__init__(bbox3D, info, ID)
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self._init_kalman_filter()
        self.kf.x[:7] = self.initial_pos.reshape((7, 1))

    def _init_kalman_filter(self):
        """Initialize the Kalman Filter matrices."""
        # State transition matrix
        self.kf.F = np.eye(10)
        self.kf.F[0, 7] = self.kf.F[1, 8] = self.kf.F[2, 9] = 1
        """ self.kf.F
           [[1. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
            [0. 1. 0. 0. 0. 0. 0. 0. 1. 0.]
            [0. 0. 1. 0. 0. 0. 0. 0. 0. 1.]
            [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
        """
        # Measurement function
        self.kf.H = np.zeros((7, 10))
        np.fill_diagonal(self.kf.H, 1)
        """ self.kf.H
           [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]
        """
        # Initial state uncertainty
        self.kf.P *= 10.
        self.kf.P[7:, 7:] *= 1000.

        # Process uncertainty
        self.kf.Q[7:, 7:] *= 0.01

    def predict(self):
        """
        卡曼滤预测
        主要yaw的值应属于(-pi,pi)
        """

        self.kf.predict()

    # def update(self, bbox3D: np.ndarray):
    #     """Update the state with the new measurement."""
    #     self.kf.update(bbox3D.reshape((7, 1)))
    #     # self.kf.x[6] = bbox3D[6]

    def update(self, bbox3D: np.ndarray):
        """根据新检测框更新卡尔曼滤波状态,处理yaw周期性问题"""
        # 获取当前的卡尔曼滤波器的 yaw 值
        previous_yaw = self.kf.x[6, 0]
        new_yaw = bbox3D[6]

        # 计算角度差，确保差值在 [-pi, pi] 范围内
        yaw_diff = new_yaw - previous_yaw
        yaw_diff = limit_angle(yaw_diff)

        # 更新 yaw 值，保持平滑过渡
        bbox3D[6] = previous_yaw + yaw_diff

        # 调用卡尔曼滤波器的更新方法
        self.kf.update(bbox3D.reshape((7, 1)))

        # 确保更新后的 yaw 在 [-pi, pi] 范围内
        self.kf.x[6] = limit_angle(self.kf.x[6])

    def get_state(self) -> np.ndarray:
        """Return the current state estimate."""
        return self.kf.x

    def get_velocity(self) -> np.ndarray:
        """Return the object velocity in the state."""
        return self.kf.x[7:]


def associate_detections_to_trackers(detections, trackers, iou_threshold=-0.1):
    """
    Assigns detections to tracked objects (both represented as bounding boxes).
    Returns 3 lists: matches, unmatched_detections, and unmatched_trackers.

    Parameters:
        detections (list of np.array): List of detection bounding boxes.
        trackers (list of np.array): List of tracker bounding boxes.
        iou_threshold (float): Minimum IoU required for a match.

    Returns:
        matches (np.array): Array of matched indices.
        unmatched_detections (np.array): Array of unmatched detection indices.
        unmatched_trackers (np.array): Array of unmatched tracker indices.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 7), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            giou,iou3d,iou2d = iou.calculate_iou(det[:7],trk[:7])
            # print(f"giou: {giou}")
            iou_matrix[d, t] = giou

    # print(f"iou_matrix : \n {iou_matrix}")

    # Perform Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    # matched_indices = np.array(list(zip(row_ind, col_ind))) #zip开销大
    matched_indices = np.stack((row_ind, col_ind), axis=1)    #np是在c级别操作,速度快

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# Example usage
if __name__ == "__main__":
    # Example bounding boxes: [x,y,z,l,w,h,yaw]
    
    detections = [np.array([1, 1, 1, 2, 2, 2, 0]), np.array([2, 2, 2, 2, 2, 2, 0])]
    trackers = [KF(np.array([1, 1, 1, 2, 2, 2, 0]), {}, 1), KF(np.array([2, 2, 2, 2, 2, 2, 0]), {}, 2)]

    # Predict the next state for each tracker
    for tracker in trackers:
        print(f"track Id:{tracker.id}")
        print(f"before predict :{tracker.kf.x.reshape((-1))}")
        tracker.predict()
        print(f"after predict :{tracker.kf.x.reshape((-1))}")

    # Get the predicted states
    predicted_states = [tracker.get_state()[:7].flatten() for tracker in trackers]

    # Associate detections to trackers
    matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, predicted_states)

    # Update matched trackers with assigned detections
    for match in matches:
        trackers[match[1]].update(detections[match[0]])
    # Handle unmatched detections (create new trackers) and unmatched trackers
    print("Matches: \n", matches)
    print("Unmatched Detections:", unmatched_detections)
    print("Unmatched Trackers:", unmatched_trackers)
