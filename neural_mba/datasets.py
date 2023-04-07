"""
libary for generating random samples for a MBA expression.
"""

import random
from typing import Any, Iterator, Tuple
import torch
from torch import Tensor
from torch.utils.data import  Dataset

class MBADataset(Dataset[Any]):
    """
    Dataset wrapper to generate random samples for a MBA expression.
    """

    def __init__(self, expr: str, num_samples: int, device: str = "cuda:0"):
        super().__init__()
        self.num_samples = num_samples
        self.expr = expr
        self.samples = list(self.sample())
        self.device = device

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        args, res = self.samples[idx]
        return args.to(self.device), res.to(self.device)

    def sample(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """generates random samples for the expression"""
        for _ in range(self.num_samples):
            x = random.randint(0, 2**8-1)
            y = random.randint(0, 2**8-1)
            res = float(self.eval_expr(self.expr, x, y))
            # type: ignore
            yield torch.tensor([float(x), float(y)]), torch.tensor([res])

    def eval_expr(self, expr_str: str, x_val: int, y_yal: int) -> Any:
        """returns the evaluation of the expression"""
        return eval(expr_str) # pylint: disable=eval-used
