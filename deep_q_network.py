import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim=5, output_dim=3):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.LayerNorm(64)
        self.fc4 = nn.Linear(64, output_dim)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.silu(self.bn1(self.fc1(x)))   # SiLU激活
        x = F.silu(self.bn2(self.fc2(x)))
        x = F.silu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
