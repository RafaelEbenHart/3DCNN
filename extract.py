# extract.py
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def extract_skeleton(image_bgr):
    """
    Input  : BGR image (OpenCV)
    Output : numpy array (33,3) normalized landmarks OR None
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks.landmark

    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    return points  # (33, 3)
