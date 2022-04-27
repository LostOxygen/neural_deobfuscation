"""main file to run the lda approximation"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import time
import socket
import datetime
import argparse
import os
import torch
import numpy as np

from neural_mba.train import train

torch.backends.cudnn.benchmark = True


def main() -> None:
    """main function for lda stability testing"""
    start = time.perf_counter()
    print("\n\n\n"+"#"*55)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {torch.get_num_threads()} CPU cores with "
          f"{os.cpu_count()} threads and "
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print("#"*55)
    print()

    # ---------------- Train MBA Model ----------------
    train()

    # ---------------- Train Mapping Model ----------------

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))