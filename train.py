# train.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# tambahkan src ke sys.path supaya import berjalan
repo_path = os.path.join(os.path.dirname(__file__), "transferModel", "src")
if repo_path not in sys.path:
    sys.path.append(repo_path)

from dataloader import SkeletonDataset
from modelGesture import GestureTCN

# ===============================
# Dataset
# ===============================
dataset = SkeletonDataset("data/gesture")

# Split train / test (80% / 20%)
test_ratio = 0.2
test_size = int(len(dataset) * test_ratio)
train_size = len(dataset) - test_size

train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# ===============================
# Device
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Model
# ===============================
model = GestureTCN(num_joints=33, in_features=3, num_classes=len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# ===============================
# Training + Testing per Epoch
# ===============================
EPOCHS = 200

for epoch in range(EPOCHS):

    # -------- TRAIN --------
    model.train()
    train_loss = 0

    for x, y in train_loader:
        # x: (B, T, J, C)
        x = x.permute(0, 3, 1, 2).unsqueeze(-1).to(device)  # (B, C, T, V, M)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------- TEST --------
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.permute(0, 3, 1, 2).unsqueeze(-1).to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    acc = correct / total if total > 0 else 0

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Test Loss: {test_loss:.4f} | "
          f"Test Acc: {acc*100:.2f}%")

# ===============================
# Save model
# ===============================
torch.save(model.state_dict(), "gesture_tcn.pth")
print("Model saved: gesture_tcn.pth")
