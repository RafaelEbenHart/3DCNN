# train.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# tambahkan src ke sys.path supaya import berjalan

repo_path = os.path.join(os.path.dirname(__file__), "transferModel", "src")
if repo_path not in sys.path:
    sys.path.append(repo_path)


from dataloader import SkeletonDataset
from modelGesture import GestureTCN

# --- Dataset & Dataloader ---
dataset = SkeletonDataset("data/gesture")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model ---
model = GestureTCN(num_joints=33, in_features=3, num_classes=len(dataset.classes))
model = model.to(device)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# --- Training Loop ---
EPOCHS = 1000

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        x = x.permute(0,3,1,2).unsqueeze(-1).to(device)  # (B, C, T, V, M)
        y = torch.tensor(y, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# --- Simpan model ---
torch.save(model.state_dict(), "gesture_tcn.pth")
print("Model saved: gesture_tcn.pth")
