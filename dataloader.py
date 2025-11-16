import torch
import numpy as np
from torch.utils.data import Dataset
import os

class SkeletonDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for idx, cls in enumerate(self.classes):
            class_path = os.path.join(root_dir, cls)
            for f in os.listdir(class_path):
                if f.endswith('.npy'):
                    self.samples.append(os.path.join(class_path, f))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = np.load(self.samples[idx])         # (T, J, C)
        x = torch.tensor(x, dtype=torch.float32)
        return x, self.labels[idx]
