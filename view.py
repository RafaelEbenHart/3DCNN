import numpy as np
import cv2
import mediapipe as mp

# ======================================================
# SET FILE .NPY DI SINI
# ======================================================
GESTURE = "absolute"                      # <-- ubah sesuai kebutuhan
TYPE = "gesture"
NUMBER = "007"
NPY_PATH = f"data/{TYPE}/{GESTURE}/sample_{NUMBER}.npy"   # <-- ubah sesuai kebutuhan
DELAY = 100                                   # delay antar frame (ms)
# ======================================================

# -----------------------------
# LOAD DATASET .NPY
# -----------------------------
data = np.load(NPY_PATH)  # shape (T, J, C)

print("File         :", NPY_PATH)
print("Shape        :", data.shape)
print("Frame count  :", data.shape[0])
print("Joints       :", data.shape[1])
print("Channels     :", data.shape[2])
print("--------------------------------------")

# -----------------------------
# MEDIAPIPE DRAWING UTILS
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
connections = mp_pose.POSE_CONNECTIONS   # sambungan antar joint

# -----------------------------
# NORMALISASI JADI PIXEL
# -----------------------------
def to_pixel(x, y, w, h):
    return int(x * w), int(y * h)

# -----------------------------
# VISUALISASI
# -----------------------------
W, H = 640, 480

for idx, frame in enumerate(data):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    joints = {}

    # gambar joint
    for jid, (x, y, z) in enumerate(frame):
        px, py = to_pixel(x, y, W, H)
        joints[jid] = (px, py)
        cv2.circle(canvas, (px, py), 4, (255, 255, 255), -1)

    # gambar koneksi pose
    for c in connections:
        if c[0] in joints and c[1] in joints:
            cv2.line(canvas, joints[c[0]], joints[c[1]], (0, 255, 0), 2)

    # label frame
    cv2.putText(canvas, f"Frame {idx+1}/{len(data)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Skeleton Viewer", canvas)
    key = cv2.waitKey(DELAY)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
