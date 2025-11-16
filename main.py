from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import io

import mediapipe as mp
from modelGesture import GestureTCN

app = FastAPI()

# ---------------------------------------
# STATIC FOLDER
# ---------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------
# LOAD CLASSES
# ---------------------------------------
DATA_DIR = Path("data/gesture")
CLASS_NAMES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()]) \
    if DATA_DIR.exists() else ["absolute", "confused", "idea"]

NUM_CLASSES = len(CLASS_NAMES)

# Representative default images
CLASS_IMAGES = {}
for name in CLASS_NAMES:
    img_path = Path("static") / f"{name}.jpg"
    CLASS_IMAGES[name] = str(img_path) if img_path.exists() else str(Path("static/loading.gif"))

# ---------------------------------------
# LOAD MODEL
# ---------------------------------------
model = GestureTCN(num_joints=33, in_features=3, num_classes=NUM_CLASSES)
MODEL_PATH = Path("model/gesture_tcn.pth")

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("[INFO] Model loaded successfully.")
else:
    print("[WARN] Model not found — running in dummy mode")
    model = None


# ---------------------------------------
# MEDIAPIPE
# ---------------------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)


# ---------------------------------------
# EXTRACT POSE (MATCH eval_cam.py)
# ---------------------------------------
def extract_keypoints_eval(frame):
    """Extract pose identical to eval_cam.py → produces (33,3)."""

    # Convert: PIL → numpy
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    results = holistic.process(frame_rgb)

    keypoints = np.zeros((33, 3), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i < 33:
                keypoints[i] = [lm.x, lm.y, lm.z]

    return keypoints   # (33,3)


# ---------------------------------------
# SERVE FRONTEND HTML
# ---------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("static/index.html").read_text()


# ---------------------------------------
# CAMERA FRAME UPLOAD (REAL-TIME PREDICT)
# ---------------------------------------
@app.post("/upload-from-web")
async def upload_from_web(file: UploadFile = File(...)):

    img_bytes = await file.read()

    # Validate image
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        return JSONResponse({"error": "Invalid image format"}, status_code=400)

    frame = np.array(img)  # PIL → numpy

    # 1. Extract keypoints — SAME AS eval_cam.py
    keypoints = extract_keypoints_eval(frame)  # (33,3)

    # 2. Build tensor: (1, 1, 33, 3) — EXACT MATCH TO YOUR MODEL
    input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 3. Predict
    if model is None:
        pred_idx = np.random.randint(0, NUM_CLASSES)
    else:
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = int(torch.argmax(output, dim=1))

    class_name = CLASS_NAMES[pred_idx]

    return {
        "class_name": class_name,
        "image_url": f"/static/{class_name}.jpg"
    }


# ---------------------------------------
# CLASS IMAGE ENDPOINT
# ---------------------------------------
@app.get("/class-image/{class_name}")
async def class_image(class_name: str):
    if class_name not in CLASS_IMAGES:
        return JSONResponse({"error": "Unknown class"}, status_code=404)

    return FileResponse(CLASS_IMAGES[class_name])


# ---------------------------------------
# RUN FASTAPI
# ---------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
