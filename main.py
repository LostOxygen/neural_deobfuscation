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
from neural_mba.utils import create_datasets

torch.backends.cudnn.benchmark = True

DATASET_SIZE = 10000
DATA_PATH = "./data/"

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

    # ---------------- Train MBA Model --------------------
    # _ = train("x+y", "add")
    # _ = train("x-y", "sub")
    # _ = train("x*y", "mul")

    # ---------------- Create Mapping Dataset -------------
    if not os.path.isfile(DATA_PATH+"train_data.tar"):
        create_datasets(train_size=int(DATASET_SIZE*0.8), test_size=int(DATASET_SIZE*0.2))

    # ---------------- Train Mapping Model ----------------
    train_mapping(epochs=100, batch_size=32, dataset_size=int(DATASET_SIZE*0.8))

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))