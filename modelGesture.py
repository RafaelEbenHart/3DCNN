# GestureTCN.py (fixed final)
import sys
import os
import torch
import torch.nn as nn
import numpy as np

# --- Step 0: tambahkan folder src ke sys.path supaya bisa import model ---
repo_src_path = os.path.join(os.path.dirname(__file__), "transferModel", "src")
if repo_src_path not in sys.path:
    sys.path.append(repo_src_path)

# --- Graph (Mediapipe 33 joints) ---
num_nodes = 33

# edges berdasarkan struktur pose Mediapipe (lengkapi bila perlu)
edges = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(12,24),(24,23),(23,11),
    (11,13),(13,15),
    (12,14),(14,16),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
]

# ============================================================
# BANGUN MATRIX A (3-KERNEL ST-GCN) — sebagai numpy dulu
# ============================================================
A_np = np.zeros((3, num_nodes, num_nodes), dtype=np.float32)

# self links
for i in range(num_nodes):
    A_np[0, i, i] = 1

# forward (parent → child)
for p, c in edges:
    A_np[1, p, c] = 1

# backward (child → parent)
for p, c in edges:
    A_np[2, c, p] = 1

edge_np = np.ones((3, num_nodes, num_nodes), dtype=np.float32)

# ------------------------------------------------------------
# CONVERT: A harus torch.Tensor; edge biarkan None (repo expects flag)
# ------------------------------------------------------------
A = torch.tensor(A_np, dtype=torch.float32)    # shape (3, V, V)
# don't pass edge matrix (repo expects edge flag), use edge=None
edge = None

# --- Step 1: import EfficientGCN dari repo ---
from transferModel.src.model.nets import EfficientGCN

# --- GestureTCN wrapper ---
class GestureTCN(nn.Module):
    def __init__(self, num_joints=33, in_features=3, num_classes=3):
        super().__init__()

        # EfficientGCN backbone
        # IMPORTANT: set num_class=num_classes so internal classifier emits correct logits
        self.backbone = EfficientGCN(
            data_shape=(1, in_features, 30, num_joints, 1),
            block_args=[[64,1,1], [128,2,2], [256,2,2]],
            fusion_stage=1,
            stem_channel=64,
            num_class=num_classes,   # <-- use final class count here
            drop_prob=0.5,
            layer_type="Basic",
            kernel_size=(9,1),
            bias=True,
            act=nn.LeakyReLU(inplace=True),
            edge=edge,
            A=A,
            att_type="CBAM"
        )

        # DO NOT overwrite backbone.classifier.fc with nn.Linear.
        # EfficientGCN classifier expects Conv3d after GAP.

    def forward(self, x):
        # Accept shapes like:
        # (N, T, V, C) or (N, C, T, V) or (N, C, T, V, 1)
        # Normalize to (N, 1, C, T, V, 1)

        # If provided (N, C, T, V, 1)  -> make (N, 1, C, T, V, 1)
        if x.dim() == 5 and x.shape[-1] == 1 and x.shape[1] in (3,):
            x = x.unsqueeze(1)  # (N, 1, C, T, V, 1)

        # If (N, C, T, V) -> (N, 1, C, T, V, 1)
        elif x.dim() == 4 and x.shape[1] in (3,):
            x = x.unsqueeze(1).unsqueeze(-1)

        # If (N, T, V, C) -> permute -> (N, 1, C, T, V, 1)
        elif x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).unsqueeze(1).unsqueeze(-1)

        else:
            # final fallback: try to squeeze unnecessary trailing dims then normalize
            while x.dim() > 6:
                x = x.squeeze(-1)
            if x.dim() == 5 and x.shape[1] in (3,):
                x = x.unsqueeze(1)
            if x.dim() != 6:
                raise ValueError(f"Unrecognized input shape for model: {x.shape}")

        # final check
        assert x.dtype == torch.float32 or x.dtype == torch.float64, "input must be float tensor"
        assert x.dim() == 6, f"Final shape wrong: {x.shape}"

        out, feat = self.backbone(x)   # out already shape (N, num_classes)
        return out

# --- Quick local test when run as script ---
if __name__ == "__main__":
    model = GestureTCN(num_joints=33, in_features=3, num_classes=3)
    x_dummy = torch.randn(2, 3, 30, 33, 1).float()  # (B, C, T, V, 1)
    out = model(x_dummy)
    print("Output shape:", out.shape)  # (2, num_classes)
