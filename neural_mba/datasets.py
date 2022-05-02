import torch
from torch import nn
import random
from torch import Tensor
from torch.utils.data import  Dataset
from typing import Any, List, Iterator, Tuple

class MBADataset(Dataset[Any]):

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
        for _ in range(self.num_samples):
            x = random.randint(0, 2**8-1)
            y = random.randint(0, 2**8-1)
            res = float(self.eval_expr(self.expr, x, y))
            # type: ignore
            yield torch.tensor([float(x), float(y)]), torch.tensor([res])

    def eval_expr(self, expr_str: str, x: int, y: int) -> Any: # pylint: disable=unused-argument
        return eval(expr_str) # pylint: disable=eval-used


class MappingDataset(Dataset[Any]):

    def __init__(self, num_samples: int, model_list: List[nn.Sequential], device: str):
        super().__init__()
        self.num_samples = num_samples
        self.samples = list(self.sample())
        self.model_list = model_list
        self.device = device

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        args, res = self.samples[idx]
        return args.to(self.device), res.to(self.device)

    def sample(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for _ in range(self.num_samples):
            model_x = random.randint(0, 2**8-1)
            model_y = random.randint(0, 2**8-1)
            model_id = random.randint(0, len(self.model_list)-1)

            match model_id:
                case 0: # label is "add"
                    label = torch.tensor([1, 0, 0])
                case 1:  # label is "sub"
                    label = torch.tensor([0, 1, 0])
                case 2:  # label is "mul"
                    label = torch.tensor([0, 0, 1])

            # 
            model_interna = self.model_list[model_id](torch.tensor([model_x, model_y]))

            # type: ignore
            yield model_interna, label

