from locale import strcoll
import os

import torch
from torch import nn
import numpy as np

from tqdm import tqdm
import webdataset as wds

from neural_mba.train import non_verbose_train
from neural_mba.models import MappingModel

DATA_PATH = "./data/"
MODEL_PATH = "./models/"

def create_datasets(train_size: int, test_size: int, device: str) -> None:
    """
    helper function to create and save the datasets. Dataset maps the weights of the trained
    model to the corresponding operation.

    Parameters:
        train_size: the size of the training dataset
        test_size: the size of the test dataset
        device: cpu or cuda device string
    
    Returns:
        None
    """
    print("[ saving train/test data and labels ]")
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)


    with (
        wds.TarWriter(DATA_PATH+"train_data.tar") as train_sink,
        wds.TarWriter(DATA_PATH+"test_data.tar") as test_sink,
    ):
        for idx in tqdm(range(train_size)):
            match idx:
                # 33% of the train data for "add"
                case idx if idx >= 0 and idx < np.floor(train_size*0.33):
                    label = torch.LongTensor([0])
                    model = non_verbose_train("x+y", "add", device)
                    model_weights = get_model_weights(model)

                # 33% of the train data for "sub"
                case idx if idx >= np.floor(train_size*0.33) and idx < np.floor(train_size*0.66):
                    label = torch.LongTensor([1])
                    model = non_verbose_train("x-y", "sub", device)
                    model_weights = get_model_weights(model)

                # 33% of the train data for "mul"
                case idx if idx >= np.floor(train_size*0.66) and idx < train_size:
                    label = torch.LongTensor([2])
                    model = non_verbose_train("x*y", "mul", device)
                    model_weights = get_model_weights(model)

            # save the model weights and the label as a tar file
            train_sink.write({
                "__key__": "sample%06d" % idx,
                "input.pyd": model_weights,
                "output.pyd": label,
            })

        for idx in tqdm(range(test_size)):
            match idx:
                # 33% of the test data for "add"
                case idx if idx >= 0 and idx < np.floor(test_size*0.33):
                    label = torch.LongTensor([0])
                    model = non_verbose_train("x+y", "add", device)
                    model_weights = get_model_weights(model)

                # 33% of the test data for "sub"
                case idx if idx >= np.floor(test_size*0.33) and idx < np.floor(test_size*0.66):
                    label = torch.LongTensor([1])
                    model = non_verbose_train("x-y", "sub", device)
                    model_weights = get_model_weights(model)

                # 33% of the test data for "mul"
                case idx if idx >= np.floor(test_size*0.66) and idx < test_size:
                    label = torch.LongTensor([2])
                    model = non_verbose_train("x*y", "mul", device)
                    model_weights = get_model_weights(model)

            # save the model weights and the label as a tar file
            test_sink.write({
                "__key__": "sample%06d" % idx,
                "input.pyd": model_weights,
                "output.pyd": label,
            })


def get_model_weights(model: nn.Sequential) -> torch.FloatTensor:
    """
    Helper function to extract and flatten the weights of a given model into a 1D tensor.

    Parameters:
        model: nn.Sequential model to extract the weights from

    Returns:
        weights: 1D FloatTensor of the flattened weights on CPU device
    """
    weights = None
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            if weights is None:
                weights = layer.weight.data.flatten()
            else:
                weights = torch.cat((weights, layer.weight.data.flatten()), 0)

    return torch.FloatTensor(weights.cpu())


def test_model() -> None:
    """
    Helper function to test the models prediction accuracy on new random operations.

    Parameters:
        None

    Returns:
        None
    """
    operations_list = [("x+y", "add"), ("x-y", "sub"), ("x*y", "mul")]

    model = MappingModel()
    checkpoint = torch.load(MODEL_PATH + "mba_model_mapping", map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['net'], strict=False)
    model.eval()

    total_tests = 1000
    correct_tests = 0

    for i in range(total_tests):
        rand_operation = operations_list[np.random.randint(0, 3)]
        print(f"Chose operation: {rand_operation}")
        input_model = non_verbose_train(rand_operation[0], rand_operation[1], "cpu")
        input_weights = get_model_weights(input_model)
        predicted_operation = model(input_weights).argmax()
        
        if operations_list[predicted_operation][1] == rand_operation[1]:
            correct_tests += 1
            print(f"[ ✓ prediction correct -> (pred: {operations_list[predicted_operation][1]}," \
                  f" true: {rand_operation[1]}) ]\n")
        else:
            print(f"[ × prediction incorrect -> (pred: {operations_list[predicted_operation][1]}," \
                  f" true: {rand_operation[1]}) ]\n")

    print(f"\n[ test results ]\n" \
          f"{correct_tests}/{total_tests} correct predictions\n" \
          f"{(correct_tests/total_tests)*100}% accuracy\n")