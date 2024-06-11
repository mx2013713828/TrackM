import os
import argparse
import numpy as np
from trackm import KF, Filter, associate_detections_to_trackers
import iou


class TrackerManager:
    def __init__(self, max_age=5, min_hits=3):
        self.trackers = []
        self.next_id = 1
        self.max_age = max_age  # Maximum number of frames to keep a track without updates
        self.min_hits = min_hits  # Minimum hits to consider a track reliable
        self.debug = False
    def update(self, detections):
        # Predict the current state of all trackers
        for tracker in self.trackers:
            tracker.predict()  ## 需测试predict前后
        
        # If no existing trackers, initialize them
        if len(self.trackers) == 0:
            self.create_new_trackers(detections, list(range(len(detections))))
        else:
            # Get current states of all trackers
            # tracker_states = np.array([tracker.get_state().flatten() for tracker in self.trackers])
            tracker_states = [np.array(tracker.get_state().flatten()) for tracker in self.trackers]

            # Associate detections to trackers
            if self.debug:
                print(f"start matching for  tracker states:  \n {tracker_states} \n and detections : \n {detections} ")
            matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, tracker_states)
            
            # Debug output
            if self.debug:
                print(f"Matches: \n{matches}")
                print(f"Unmatched Detections: {unmatched_detections}")
                print(f"Unmatched Trackers: {unmatched_trackers}")

            # 对命中的跟踪器更新状态
            self.update_trackers(detections, matches)

            # 对未命中的检测框创建新的跟踪器
            self.create_new_trackers(detections, unmatched_detections)

            # 增加未命中tracker的age
            self.increment_age_unmatched_trackers(unmatched_trackers)

        # 原地删除长时间未命中的tracker
        self.trackers = [tracker for tracker in self.trackers if tracker.time_since_update < self.max_age]

    def get_tracks(self):
        
        # Only return tracks that have been updated at least min_hits times
        return np.array([[*tracker.get_state()[:7].flatten(), tracker.info.get('score', 0),tracker.info.get('class_id', -1), tracker.id] 
                         for tracker in self.trackers if tracker.hits >= self.min_hits])


    def create_new_trackers(self, detections, unmatched_detections):
        # Create new trackers for unmatched detections
        for idx in unmatched_detections:
            tracker_tmp = KF(detections[idx][:7], {'score': detections[idx][8],'class_id': detections[idx][7]}, self.next_id)
            self.trackers.append(tracker_tmp)
            print(f"init KF with {detections[idx][:7]} and id {self.next_id}")
            print(f"init KF state: \n{tracker_tmp.get_state().flatten()}")
            print(f"init KF velocity: \n{tracker_tmp.get_velocity().flatten()}")
            self.next_id += 1

    def update_trackers(self, detections, matches):
        # Update matched trackers with the assigned detections
        if self.debug:
            print(f"update trackers with detections :\n{detections}")
        for match in matches:
            # tracker_idx, detection_idx = match
            detection_idx, tracker_idx = match
            print(f"state before update :\n{self.trackers[tracker_idx].get_state().flatten()}")
            print(f"update using {detections[detection_idx][:7]}")
            self.trackers[tracker_idx].update(detections[detection_idx][:7])
            print(f"state after update :\n{self.trackers[tracker_idx].get_state().flatten()}")
            self.trackers[tracker_idx].hits += 1
            self.trackers[tracker_idx].time_since_update = 0
            self.trackers[tracker_idx].info['score'] = detections[detection_idx][8]
            self.trackers[tracker_idx].info['class_id'] = detections[detection_idx][7]
            print(f"get velocity :{self.trackers[tracker_idx].get_velocity().flatten()}")
    def increment_age_unmatched_trackers(self, unmatched_trackers):
        # Increase the age of unmatched trackers
        for idx in unmatched_trackers:
            self.trackers[idx].time_since_update += 1


def get_detections(detections_file):
    detections = []

    with open(detections_file, 'r') as file:
        for line in file:
            data = line.strip().split()
            detection = [float(x) for x in data[:7]] + [int(data[7]), float(data[8])]
            detections.append(np.array(detection))
    return detections
def parse_args():
    parser = argparse.ArgumentParser(description="Track objects in video frames.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to input folder containing detection results.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to output folder to save tracking results.")
    parser.add_argument('--max_age', type=int, default=5, help="Maximum number of frames to keep a track without updates.")
    parser.add_argument('--min_hits', type=int, default=3, help="Minimum hits to consider a track reliable.")
    return parser.parse_args()

def main():
    args = parse_args()

    tracker_manager = TrackerManager(max_age=args.max_age, min_hits=args.min_hits)

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Get list of input files sorted by frame index
    input_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith('.txt')])
    print("Start tracking")
    for frame_idx, input_file in enumerate(input_files):
        # Load detections from the current frame
        print(f"track for detection {input_file}")
        detection_file_path = os.path.join(args.input_folder, input_file)
        detections = get_detections(detection_file_path)  # Assuming detections are comma-separated

        # Update tracker manager with current detections
        tracker_manager.update(detections)

        # 获取当前的跟踪结果
        tracks = tracker_manager.get_tracks()
        print(f"当前第 {frame_idx} 帧的跟踪结果：{tracks}")

        # 保存当前帧的跟踪结果
        output_file_path = os.path.join(args.output_folder, f'{frame_idx:06d}.txt')
        np.savetxt(output_file_path, tracks, delimiter=' ', fmt='%f')
        if frame_idx == 10:
            break

if __name__ == "__main__":
    main()
