"""Model"""
from torch.nn import Linear, ReLU, Module, Sequential
from torch import Tensor


class MBAModel(Module):
    """MBAModel"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = Sequential(
            Linear(2, 128),
            ReLU(),
            Linear(128, 128),
            ReLU(),
            Linear(128, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x


class MappingModel(Module):
    """MappingModel"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = Sequential(
            Linear(16768, 4096),
            ReLU(),
            Linear(4096, 2048),
            ReLU(),
            Linear(2048, 1024),
            ReLU(),
            Linear(1024, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x
