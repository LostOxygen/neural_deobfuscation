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
import numpy as np

from neural_mba.train import train

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

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))
