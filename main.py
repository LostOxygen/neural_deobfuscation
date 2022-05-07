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

from neural_mba.train import train, train_mapping
from neural_mba.utils import create_datasets, test_model

torch.backends.cudnn.benchmark = True

DATASET_SIZE = 10000
DATA_PATH = "./data/"

def main(gpu: int) -> None:
    """main function for lda stability testing"""
    start = time.perf_counter()
    if gpu is None:
        device = "cpu"
    elif gpu == 0:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif gpu == 1:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"

    print("\n\n\n"+"#"*55)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {torch.get_num_threads()} CPU cores with "
          f"{os.cpu_count()} threads and "
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print(f"## Using: {device}")
    print("#"*55)
    print()

    # ---------------- Train MBA Model --------------------
    # _ = train("x+y", "add", device=device)
    # _ = train("x-y", "sub", device=device)
    # _ = train("x*y", "mul", device=device)

    # ---------------- Create Mapping Dataset -------------
    if not os.path.isfile(DATA_PATH+"train_data.tar"):
        create_datasets(train_size=int(DATASET_SIZE*0.8),
                        test_size=int(DATASET_SIZE*0.2),
                        device=device)

    # ---------------- Train Mapping Model ----------------
    train_mapping(epochs=100, batch_size=512, dataset_size=int(DATASET_SIZE*0.8), device=device)

    # ---------------- Test Mapping Model -----------------
    test_model()

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=None)
    args = parser.parse_args()
    main(**vars(args))