import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, rgb_frame):
        results = self.holistic.process(rgb_frame)
        h, w, _ = rgb_frame.shape
        
        # 1. Pose Landmarks (33 points x 4 = 132)
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(132)
            
        # 2. Left Hand Landmarks (21 points x 3 = 63)
        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        else:
            lh = np.zeros(63)
            
        # 3. Right Hand Landmarks (21 points x 3 = 63)
        rh_area = 0
        hand_roi = None
        if results.right_hand_landmarks:
            rh_coords = [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            rh = np.array(rh_coords).flatten()
            
            # Calculate hand bounding box area for the router
            x_coords = [coord[0] * w for coord in rh_coords]
            y_coords = [coord[1] * h for coord in rh_coords]
            
            x1, x2 = max(0, int(min(x_coords))-20), min(w, int(max(x_coords))+20)
            y1, y2 = max(0, int(min(y_coords))-20), min(h, int(max(y_coords))+20)
            rh_area = (x2 - x1) * (y2 - y1)
            
            if rh_area > 0 and (y2 > y1) and (x2 > x1):
                hand_roi = cv2.resize(rgb_frame[y1:y2, x1:x2], (128, 128))
        else:
            rh = np.zeros(63)

        # Build the final 258-dimensional feature vector
        keypoints = np.concatenate([pose, lh, rh])
        return keypoints, hand_roi, rh_area