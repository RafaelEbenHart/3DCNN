import torch
import torch.nn as nn

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]

            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size,
                          padding=(kernel_size - 1) * dilation,
                          dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GestureTCN(nn.Module):
    def __init__(self, num_joints=33, in_features=3, num_classes=3):
        super().__init__()

        self.input_dim = num_joints * in_features

        self.tcn = TemporalConvNet(
            num_inputs=self.input_dim,
            num_channels=[128, 256, 256]
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, T, J, C)
        B, T, J, C = x.shape
        x = x.reshape(B, T, J*C)     # (B, T, 99)
        x = x.transpose(1, 2)        # (B, 99, T)
        y = self.tcn(x)              # (B, 256, T)
        y = y.mean(dim=2)            # Global temporal pooling
        return self.fc(y)
