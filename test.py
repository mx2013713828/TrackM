import numpy as np
import iou
import trackm
import unittest
from trackm import KF, Filter, associate_detections_to_trackers

def test_box2corners():
    print("Testing box2corners function:")
    test_cases = [
        [0, 0, 0, 2, 2, 2, 0],
        [1, 1, 1, 2, 1, 1, np.pi/4],
        [-1, -1, -1, 3, 2, 1, np.pi/2]
    ]

    for i, bbox in enumerate(test_cases):
        corners = iou.box2corners(bbox)
        print(f"Test case {i+1} corners:\n{corners}\n")

def test_giou(boxa, boxb):
    print("Testing GIoU function:")

    giou,iou3d,iou2d = iou.calculate_iou(boxa,boxb)
    print(f' giou: {giou} \n iou3d: {iou3d} \n iou2d: {iou2d}')

# 测试类
class TestAssociation(unittest.TestCase):

    def test_normal_matching(self):
        detections = [np.array([1, 1, 1, 2, 2, 2, 0]), np.array([2, 2, 2, 2, 2, 2, 0])]
        trackers = [np.array([1, 1, 1, 2, 2, 2, 0]), np.array([2, 2, 2, 2, 2, 2, 0])]

        matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, trackers, iou_threshold=0.3)
        print(matches)
        self.assertTrue(len(matches) > 0)
        self.assertTrue(len(unmatched_detections) >= 0)
        self.assertTrue(len(unmatched_trackers) >= 0)

    def test_no_matching_trackers(self):
        detections = [np.array([1, 1, 1, 2, 2, 2, 0])]
        trackers = []

        matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, trackers, iou_threshold=0.3)
        
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(unmatched_detections), 1)
        self.assertEqual(len(unmatched_trackers), 0)

    def test_partial_matching(self):
        detections = [np.array([1, 1, 1, 2, 2, 2, 0]), np.array([3, 3, 3, 2, 2, 2, 0])]
        trackers = [np.array([1, 1, 1, 2, 2, 2, 0])]

        matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, trackers, iou_threshold=0.3)
        # print(matches)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(unmatched_detections), 1)
        self.assertEqual(len(unmatched_trackers), 0)


if __name__ == "__main__":
    print("Start Testing : ")
    # test_box2corners()
    # boxa = np.array([0, 0, 0, 2, 2, 2, 1.7])
    # boxb = np.array([1., 1., 0, 2, 2, 2, 1.7])
    # test_giou(boxa,boxb)
    trk = [np.array([13.70101796, 4.57136452, -0.74235851, 4.433886, 1.823255, 2., 0.54469167, 0. ,0.,0.]),np.array([ 8.7468731,  -6.28502669, -0.84568863, 0.972283, 0.767881, 1.714062, 0.32944867, 0., 0. ,0. ])]
    det = [np.array([13.87061676, 4.66908187, -0.64779444, 4.433886, 1.823255, 2., 0.55076867, 2. ,1.   ]),np.array([ 8.44675182, -6.33586244, -0.78998267, 0.972283, 0.767881, 1.714062, 0.31604367, 0., 1. ])]
    print(det)
    print(trk)

    matches, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(det, trk, iou_threshold=0.3)

    print(f"Matches: \n{matches}")
    print(f"Unmatched Detections: {unmatched_detections}")
    print(f"Unmatched Trackers: {unmatched_trackers}")    
    # unittest.main()
    # detection = np.array([13.701,4.57136,-0.742359,4.43389,1.82326,2,0.544692,0,0,0])
    # track = KF(detection[:7], {'score': detection[8],'class_id': detection[7]}, 1)

    # state = track.get_state().flatten()
    # print(f"init state: {state}")

    # track.predict()
    # predicted_state = track.get_state().flatten()
    # print(f"predicted state: {predicted_state}")

    # new_detection = np.array([13.8770, 4.66908, -0.64779, 4.433886, 1.823255, 2.0, 0.55076867 ])
    # track.update(new_detection)
    
    # print(f"updated state: {track.get_state().flatten()}")
