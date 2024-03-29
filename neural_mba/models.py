"""Model library"""
import torch.nn as nn
from torch import Tensor


class MBAModel(nn.Module):
    """MBAModel"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_val: Tensor) -> Tensor:
        """forward pass method"""
        x_val = self.layers(x_val)
        return x_val


class MappingModel(nn.Module):
    """MappingModel"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # nn.Linear(16768, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            # nn.Linear(4096, 2048),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            # nn.Linear(2048, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            # nn.Linear(1024, 512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            # nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            # nn.Linear(128, 3),
            # nn.Softmax(dim=1)
            nn.Linear(16768, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_val: Tensor) -> Tensor:
        """forward pass method"""
        x_val = self.layers(x_val)
        return x_val
