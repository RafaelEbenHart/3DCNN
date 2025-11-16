import cv2
import torch
import numpy as np
from pathlib import Path
from model import GestureTCN
import mediapipe as mp

# ----------------------------
# MODEL CONFIG
# ----------------------------
DATA_DIR = Path("data/gesture")
CLASS_NAMES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()]) if DATA_DIR.exists() else ["absolute","confused","idea"]
NUM_CLASSES = len(CLASS_NAMES)

model = GestureTCN(num_joints=33, in_features=3, num_classes=NUM_CLASSES)
model_path = Path("gesture_tcn.pth")
if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("✅ Model loaded successfully.")
else:
    print("⚠️ Model not found. Predictions will be random.")

# ----------------------------
# POSE ESTIMATION
# ----------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

def extract_keypoints(frame):
    """
    Extract 33 keypoints (x,y,z) from frame using MediaPipe Holistic.
    Output: tensor (1,1,33,3) -> sesuai model
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    keypoints = np.zeros((33,3), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i >= 33:
                break
            keypoints[i] = [lm.x, lm.y, lm.z]

    tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,33,3)
    return tensor

# ----------------------------
# OPEN CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("Gesture Prediction", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Prediction", 1280, 720)

# ----------------------------
# PREDICTION SMOOTHING
# ----------------------------
pred_buffer = []
BUFFER_SIZE = 5
pred_class = "Detecting..."

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Ekstrak keypoints
    input_tensor = extract_keypoints(frame)

    # Prediksi model
    with torch.no_grad():
        if model_path.exists():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_buffer.append(pred_idx)
        else:
            pred_buffer.append(np.random.randint(0, NUM_CLASSES))  # random prediction

    # Simpan buffer max BUFFER_SIZE
    if len(pred_buffer) > BUFFER_SIZE:
        pred_buffer.pop(0)

    # Prediksi mayoritas dari buffer
    pred_class = CLASS_NAMES[max(set(pred_buffer), key=pred_buffer.count)]

    # Tampilkan prediksi di frame
    cv2.putText(frame, f"Prediction: {pred_class}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Gesture Prediction", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
