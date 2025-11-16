import numpy as np
import mediapipe as mp
import cv2
import os

# === Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# buat folder yang benar dan konfirmasi
os.makedirs("data/testGesture/confused", exist_ok=True)
print("Folder created / exists:", os.path.abspath("data/testGesture/confused"))

sequence = []    # menyimpan 30 frame keypoints
sample_id = 1    # nama file nantinya

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # --- Jika ada pose terdeteksi ---
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Ambil 33 keypoints (x, y, visibility)
        frame_keypoints = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark]
        )

        # Masukkan ke sequence
        sequence.append(frame_keypoints)

        # Jika sudah 30 frame → simpan file npy
        if len(sequence) == 30:
            filename = f"data/testGesture/confused/sample_{sample_id:03d}.npy"
            np.save(filename, np.array(sequence))
            print("Saved:", filename)

            sequence = []       # reset untuk sequence berikutnya
            sample_id += 1

    # --- Tampilkan ---
    cv2.imshow("Pose Estimation", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk exit
        break

cap.release()
cv2.destroyAllWindows()
