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
