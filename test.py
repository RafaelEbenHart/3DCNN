import torch
from torch.utils.data import DataLoader
from dataloader import SkeletonDataset
from modelGesture import GestureTCN
from pathlib import Path
import torch.nn as nn

DATA_DIR = Path("data/testGesture")
NUM_CLASSES = 3
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "gesture_tcn.pth"

model = GestureTCN(num_joints=33, in_features=3, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

val_dataset = SkeletonDataset(DATA_DIR)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.CrossEntropyLoss()

total_loss = 0.0
total_correct = 0
total_samples = 0

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        y = torch.tensor(y).to(DEVICE)

        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

val_loss = total_loss / total_samples
val_acc = total_correct / total_samples

print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")
