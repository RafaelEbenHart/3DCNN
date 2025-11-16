from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import io

from model import GestureTCN

app = FastAPI()


# STATIC FILES

app.mount("/static", StaticFiles(directory="static"), name="static")


# LOAD CLASSES

DATA_DIR = Path("data/gesture")
CLASS_NAMES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()]) if DATA_DIR.exists() else ["absolute", "confused", "idea"]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_IMAGES = {}
for name in CLASS_NAMES:
    img_path = Path("data/class") / f"{name}.jpg"
    CLASS_IMAGES[name] = str(img_path) if img_path.exists() else str(Path("static") / "placeholder.jpg")


# LOAD MODEL (optional)

model = GestureTCN(num_joints=33, in_features=3, num_classes=NUM_CLASSES)
MODEL_PATH = Path("model/gesture_tcn.pth")

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
else:
    model = None
    print("Model not found — running in dummy mode")


# SERVE HTML

@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("static/index.html").read_text()


# POSE EXTRACTION (DUMMY)

@app.post("/extract-pose")
async def extract_pose(file: UploadFile = File(...)):
    img_bytes = await file.read()

    if not img_bytes:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    try:
        Image.open(io.BytesIO(img_bytes))   # validate image
    except:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    keypoints = np.random.rand(33, 3).tolist()

    return {"keypoints": keypoints}


# FULL UPLOAD ENDPOINT (the one HTML uses)

@app.post("/upload-from-web")
async def upload_from_web(file: UploadFile = File(...)):

    img_bytes = await file.read()

    if not img_bytes:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    # Validate image
    try:
        Image.open(io.BytesIO(img_bytes))
    except:
        return JSONResponse({"error": "Invalid image format"}, status_code=400)

    # Dummy random prediction
    pred_idx = np.random.randint(0, len(CLASS_NAMES))
    class_name = CLASS_NAMES[pred_idx]

    return {
        "class_name": class_name,
        "image_url": f"/static/{class_name}.jpg"
    }


# GET IMAGE FOR CLASS

@app.get("/class-image/{class_name}")
async def class_image(class_name: str):
    if class_name not in CLASS_IMAGES:
        return JSONResponse({"error": "Unknown class"}, status_code=404)
    return FileResponse(CLASS_IMAGES[class_name])



# RUN

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
