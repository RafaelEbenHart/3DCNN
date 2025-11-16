import torch
import numpy as np
from modelGesture import GestureTCN
from pathlib import Path

DATA_DIR = Path("data/testGesture")
CLASS_NAMES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()]) if DATA_DIR.exists() else ["absolute","confused","idea"]

model = GestureTCN(num_joints=33, in_features=3, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("gesture_tcn.pth", map_location="cpu"))
model.eval()

# contoh pemakaian: pilih file sample dari salah satu folder kelas
sample = np.load(f"data/testGesture/{CLASS_NAMES[1]}/sample_010.npy")
sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # (1, T, J, C)

with torch.no_grad():
    output = model(sample)
    pred_idx = torch.argmax(output, dim=1).item()

print(f"Predicted class: {CLASS_NAMES[pred_idx]} (index {pred_idx})")
