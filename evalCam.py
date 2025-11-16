import cv2
import torch
import numpy as np
from pathlib import Path
from modelGesture import GestureTCN
import mediapipe as mp

# ----------------------------
# MODEL CONFIG
# ----------------------------
DATA_DIR = Path("data/gesture")
CLASS_NAMES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()]) \
    if DATA_DIR.exists() else ["absolute", "confused", "idea"]
NUM_CLASSES = len(CLASS_NAMES)

model = GestureTCN(num_joints=33, in_features=3, num_classes=NUM_CLASSES)
model_path = Path("gesture_tcn.pth")
if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully.")
else:
    print("Model not found. Predictions will be random.")

# ----------------------------
# LOAD REPRESENTATIVE IMAGES
# ----------------------------
REP_IMAGES = {}
for cname in CLASS_NAMES:
    img_path = Path(f"static/{cname}.jpg")
    if img_path.exists():
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (350, 350))  # ukuran fix window preview
        REP_IMAGES[cname] = img
    else:
        REP_IMAGES[cname] = None
        print(f"[WARN] Representative image not found for class: {cname}")

# ----------------------------
# POSE ESTIMATION
# ----------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    keypoints = np.zeros((33, 3), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i >= 33:
                break
            keypoints[i] = [lm.x, lm.y, lm.z]

    tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor  # (1,1,33,3)

# ----------------------------
# OPEN CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Feed", 1280, 720)

cv2.namedWindow("Class Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Class Preview", 400, 400)

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

    # Extract keypoints
    input_tensor = extract_keypoints(frame)

    # Predict
    with torch.no_grad():
        if model_path.exists():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_buffer.append(pred_idx)
        else:
            pred_buffer.append(np.random.randint(0, NUM_CLASSES))

    # Keep buffer size small
    if len(pred_buffer) > BUFFER_SIZE:
        pred_buffer.pop(0)

    # Majority vote
    pred_class = CLASS_NAMES[max(set(pred_buffer), key=pred_buffer.count)]

    # --------------------------------
    # SHOW REPRESENTATIVE IMAGE IN SEPARATE WINDOW
    # --------------------------------
    rep_img = REP_IMAGES.get(pred_class)
    if rep_img is not None:
        cv2.imshow("Class Preview", rep_img)

    # --------------------------------
    # CAMERA FEED (NO PREDICTION TEXT)
    # --------------------------------
    cv2.imshow("Camera Feed", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
