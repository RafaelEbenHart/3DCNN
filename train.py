import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataloader import SkeletonDataset
from model import GestureTCN

dataset = SkeletonDataset("data/gesture")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GestureTCN(num_joints=33, in_features=3, num_classes=len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 1000

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), torch.tensor(y).to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "gesture_tcn.pth")
print("Model saved: gesture_tcn.pth")
