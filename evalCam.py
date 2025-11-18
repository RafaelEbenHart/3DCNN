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
    print("Model not found — predictions will be random.")

# ----------------------------
# LOAD REPRESENTATIVE IMAGES
# ----------------------------
REP_IMAGES = {}
for cname in CLASS_NAMES:
    img_path = Path(f"static/{cname}.jpg")
    if img_path.exists():
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (350, 350))  # ukuran fix preview
        REP_IMAGES[cname] = img
    else:
        REP_IMAGES[cname] = None
        print(f"[WARN] Representative image missing for: {cname}")

# ----------------------------
# MEDIAPIPE
# ----------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# FRAME BUFFER (T = 30)
# ----------------------------
MAX_FRAMES = 30
frame_buffer = []

# ----------------------------
# FUNCTION: Extract keypoints
# ----------------------------
def extract_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    pts = np.zeros((33, 3), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:33]):
            pts[i] = [lm.x, lm.y, lm.z]

    pts = (pts - 0.5) * 2.0  # normalization

    return pts   # (33,3)

# ----------------------------
# OPEN CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Feed", 1280, 720)

cv2.namedWindow("Class Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Class Preview", 400, 400)

# Prediction smoothing
pred_buffer = []
PRED_SMOOTH = 5


# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    pts = extract_keypoints(frame)

    frame_buffer.append(pts)
    if len(frame_buffer) > MAX_FRAMES:
        frame_buffer.pop(0)

    # Predict only when 30 frames ready
    if len(frame_buffer) == MAX_FRAMES:
        seq = np.stack(frame_buffer)  # (30,33,3)
        seq = torch.tensor(seq, dtype=torch.float32)
        seq = seq.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
        # (1, 3, 30, 33, 1)

        with torch.no_grad():
            output = model(seq)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_buffer.append(pred_idx)

    else:
        pred_buffer.append(-1)

    if len(pred_buffer) > PRED_SMOOTH:
        pred_buffer.pop(0)

    if pred_buffer[-1] == -1:
        pred_class = "Detecting..."
    else:
        pred_class = CLASS_NAMES[max(set(pred_buffer), key=pred_buffer.count)]

    # ============= DISPLAY PREVIEW IMAGE =============
    rep_img = REP_IMAGES.get(pred_class)
    if rep_img is not None:
        cv2.imshow("Class Preview", rep_img)

    # ============= CAMERA FEED =============
    cv2.putText(frame, f"Prediction: {pred_class}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
