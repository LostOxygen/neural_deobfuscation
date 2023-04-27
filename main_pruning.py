"""main file to run the neural deobfuscation"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
from typing import Final
import time
import socket
import datetime
import argparse
import os
import torch
from torch.nn.utils import prune
import numpy as np

from neural_mba.train import train, test_pruning

torch.backends.cudnn.benchmark = True

COMPLEX_FUNCTION: Final[str] = "((- x) + (- ((- x) + ((- x) + (- y)))))"
SIMPLE_FUNCTION: Final[str] = "x + y"

def main(gpu: int) -> None:
    """main function for neural deobfuscation"""
    start = time.perf_counter()
    if gpu == -1 or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = f"cuda:{gpu}"

    print("\n\n\n"+"#"*55)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {torch.get_num_threads()} CPU cores with "
          f"{os.cpu_count()} threads and "
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print(f"## Using: {device}")
    print("#"*55)
    print()

    # ---------------- Train MBA Model --------------------
    model = train(expr=COMPLEX_FUNCTION, operation_suffix="complex", verbose=True, device=device)

    # prune the model
    parameters = []
    for module in model.layers.modules():
        if isinstance(module, torch.nn.Linear):
            parameters.append((module, "weight"))

    prune.global_unstructured(parameters,
                              pruning_method=prune.L1Unstructured,
                              amount=0.2)

    test_pruning(model=model, expr=COMPLEX_FUNCTION, device=device)

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))
