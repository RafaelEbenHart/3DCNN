import numpy as np
import mediapipe as mp
import cv2
import os

# =========================
# Setup Mediapipe
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# =========================
# Konfigurasi dataset
# =========================
GESTURE_NAME = "netral"  # ganti sesuai gesture
DATATYPE = "gesture"   # "gesture" atau "testGesture"
DATA_DIR = f"data/{DATATYPE}/{GESTURE_NAME}"
os.makedirs(DATA_DIR, exist_ok=True)
print("Folder created / exists:", os.path.abspath(DATA_DIR))

sequence = []        # menyimpan frame keypoints
sample_id = 1        # nama file
FRAME_COUNT = 30     # jumlah frame per sample
SMOOTH_WINDOW = 3    # untuk smoothing

# =========================
# Helper functions
# =========================
def relative_coordinates(seq):
    """Ubah keypoints jadi relatif terhadap root joint (0)"""
    return [frame - frame[0] for frame in seq]

def smooth_sequence(seq, window=3):
    """Smoothing moving average sederhana"""
    smoothed = []
    for i in range(len(seq)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(seq[start:i+1], axis=0))
    return smoothed

def pad_or_crop_sequence(seq, target_len=30):
    """Pastikan selalu target_len frame"""
    if len(seq) < target_len:
        while len(seq) < target_len:
            seq.append(seq[-1])
    elif len(seq) > target_len:
        seq = seq[-target_len:]
    return seq

# =========================
# Main loop perekaman
# =========================
while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Ambil 33 keypoints (x, y, visibility)
        frame_keypoints = np.array([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark])
        sequence.append(frame_keypoints)

        # Smoothing dan relative coordinates
        seq_proc = relative_coordinates(sequence)
        seq_proc = smooth_sequence(seq_proc, window=SMOOTH_WINDOW)

        # Jika sudah cukup frame, simpan
        if len(seq_proc) >= FRAME_COUNT:
            seq_proc = pad_or_crop_sequence(seq_proc, target_len=FRAME_COUNT)
            filename = os.path.join(DATA_DIR, f"sample_{sample_id:03d}.npy")
            np.save(filename, np.array(seq_proc))
            print("Saved:", filename)

            sequence = []       # reset sequence untuk sample berikut
            sample_id += 1

    # Tampilkan
    cv2.imshow("Pose Estimation", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk exit
        break

cap.release()
cv2.destroyAllWindows()
