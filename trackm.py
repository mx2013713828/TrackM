import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import iou
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

        # Measurement function
        self.kf.H = np.zeros((7, 10))
        np.fill_diagonal(self.kf.H, 1)

        # Initial state uncertainty
        self.kf.P *= 10.
        self.kf.P[7:, 7:] *= 1000.

        # Process uncertainty
        self.kf.Q[7:, 7:] *= 0.01

    def predict(self):
        """Predict the next state."""
        self.kf.predict()

    def update(self, bbox3D: np.ndarray):
        """Update the state with the new measurement."""
        self.kf.update(bbox3D.reshape((7, 1)))

    def get_state(self) -> np.ndarray:
        """Return the current state estimate."""
        return self.kf.x

    def get_velocity(self) -> np.ndarray:
        """Return the object velocity in the state."""
        return self.kf.x[7:]

def iou3d(bbox1, bbox2):
    """
    Compute the 3D Intersection over Union (IoU) between two 3D bounding boxes.
    This function needs to be implemented.
    """
    # Placeholder for actual IoU calculation
    return np.random.rand()

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes).
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 7), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou3d(det, trk)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*matched_indices)))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

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
    
    detections = [np.array([0, 0, 0, 0, 1, 1, 1]), np.array([1, 1, 1, 0, 1, 1, 1])]
    trackers = [KF(np.array([0, 0, 0, 0, 1, 1, 1]), {}, 1), KF(np.array([2, 2, 2, 0, 1, 1, 1]), {}, 2)]

    # Predict the next state for each tracker
    for tracker in trackers:
        tracker.predict()

    # Get the predicted states
    predicted_states = [tracker.get_state()[:7].flatten() for tracker in trackers]

    # Associate detections to trackers
    matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, predicted_states)

    # Update matched trackers with assigned detections
    for match in matches:
        trackers[match[1]].update(detections[match[0]])

    # Handle unmatched detections (create new trackers) and unmatched trackers
    print("Matches:", matches)
    print("Unmatched Detections:", unmatched_detections)
    print("Unmatched Trackers:", unmatched_trackers)
