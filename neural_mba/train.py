import os
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from typing import Any, Dict
from tqdm import trange
import numpy as np
import webdataset as wds
import pkbar

from neural_mba.datasets import MBADataset
from neural_mba.models import MBAModel, MappingModel


MODEL_PATH = "./models/"
DATA_PATH = "./data/"

TRAIN_CONFIG: Dict[str, Any] = {
    "training_samples" : 10000,
    "device" : "cuda",
    "epochs" : 3,
    "weight_decay" : 1e-4,
    "learning_rate": 0.001,
    "batch_size" : 8,
}
TEST_CONFIG: Dict[str, Any] = {
    "batch_size" : 1,
    "device" : "cuda",
    "samples" : 100,
}


def save_model(net: nn.Sequential, operation_suffix: str) -> None:
    """
    helper function which saves the given net in the specified path.
    if the path does not exists, it will be created.

    Parameters:
        net: the model to save
        operation_suffix: the suffix of the operation to create the save path with
    
    Returns:
        None
    """
    state = {
        "net": net.state_dict()
    }
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(state, f"{MODEL_PATH}mba_model_{operation_suffix}")


def adjust_learning_rate(optimizer, epoch: int, epochs: int, learning_rate: int) -> None:
    """
    helper function to adjust the learning rate
    according to the current epoch to prevent overfitting.
    
    Parameters:
        optimizer: the optimizer to adjust the learning rate with
        epoch: the current epoch
        epochs: the total number of epochs
        learning_rate: the learning rate to adjust

    Returns:
        None
    """
    new_lr = learning_rate
    if epoch >= np.floor(epochs*0.5):
        new_lr /= 10
    if epoch >= np.floor(epochs*0.75):
        new_lr /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def get_mapping_loaders(batch_size: int) -> DataLoader:
    """
    Helper function to create the dataloader for network weights to operation mapping.

    Parameters:
        batch_size: the batch size to use for the mapping dataloaders
    
    Returns:
        train_loader: the dataloader for the training
        test_loader: the dataloader for the test
    """
    train_data_path = DATA_PATH + "train_data.tar"
    test_data_path = DATA_PATH + "test_data.tar"

    # build a wds dataset, shuffle it, decode the data and create dense tensors from sparse ones
    train_dataset = wds.WebDataset(train_data_path).shuffle(1000).decode().to_tuple("input.pyd",
                                                                                    "output.pyd")
    test_dataset = wds.WebDataset(test_data_path).shuffle(1000).decode().to_tuple("input.pyd",
                                                                                  "output.pyd")

    train_loader = DataLoader((train_dataset.batched(batch_size)), batch_size=None, num_workers=0)
    test_loader = DataLoader((test_dataset.batched(batch_size)), batch_size=None, num_workers=0)

    return train_loader, test_loader


def get_loaders(expr: str, device: str) -> DataLoader:
    """
    Helper function to create the dataloader for the training and the test.

    Parameters:
        expr: the operation expression to train the model on
        device: cpu or cuda device string
    
    Returns:
        train_loader: the dataloader for the training
        test_loader: the dataloader for the test
    """
    train_dataset = MBADataset(expr, \
                     TRAIN_CONFIG["training_samples"], \
                               device=device)
    train_loader = DataLoader(train_dataset,
                    batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,
                    drop_last=True)
    test_dataset = MBADataset(expr,
                        TEST_CONFIG["samples"], \
                        device=device)
    test_loader = DataLoader(test_dataset,
                    batch_size=TEST_CONFIG["batch_size"], shuffle=True,
                    drop_last=True)

    return train_loader, test_loader


def train(expr: str, operation_suffix: str, verbose: bool, device: str) -> None:
    """
    Main function to train the model with the specified parameters. Saves the model in every
    epoch specified in SAVE_EPOCHS. Prints the model status during the training.

    Parameters:
        expr: the operation expression to train the model on
        operation_suffix: the suffix of the operation to create the save path with
        verbose: if True, prints the training status
        device: cpu or cuda device string

    Returns:
        model: trained pytorch model (but also saves the model in the specified path)
    """
    # input dimension of the model is the length of the dictionary
    model = MBAModel().to(device)
    if verbose:
        summary(model, input_size=(2,))

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"], \
                           weight_decay=TRAIN_CONFIG["weight_decay"])
    train_loader, test_loader = get_loaders(expr, device)

    with trange(1, TRAIN_CONFIG["epochs"]+1, bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        losses = [0.0]
        accs = [0]

        for epoch in pbar:
            # train for one epoch
            loss_sum = 0.0
            for batch_idx, (x, y) in enumerate(train_loader):
                if verbose:
                    pbar.set_description(f"epoch {epoch:>3} batch {batch_idx:>3}/" \
                                        f"{(len(train_loader)//TRAIN_CONFIG['batch_size'])-1:>3}" \
                                        f"loss {loss_sum/max(TRAIN_CONFIG['batch_size']*batch_idx,1):10.5f}" \
                                        f" acc xx.xx")
                pred = model.forward(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            # evaluate performance
            if verbose:
                pbar.set_description(f"epoch {epoch:>3} EVAL")
            model.eval()
            with torch.no_grad():
                no_correct = 0
                for test_batch_idx, (x, y) in enumerate(test_loader):
                    pred = model.forward(x)
                    no_correct += torch.sum(torch.round(pred) == y).item()
                    if verbose:
                        pbar.set_description(f"epoch {epoch:>3} EVAL batch {test_batch_idx:>3}/" \
                                            f"{(len(test_loader)//TEST_CONFIG['batch_size'])-1:>3}")

            model.train()
            losses += [ loss_sum / len(train_loader) ]
            accs += [ no_correct / len(test_loader) * 100 ]

            if verbose:
                pbar.set_description(f"epoch {epoch:>3} batch {batch_idx:>3}/" \
                                    f"{(len(train_loader)//TRAIN_CONFIG['batch_size'])-1:>3} " \
                                    f" loss {losses[-1]:10.5f} acc {accs[-1]:5.2f}")
                print()

        for idx, (x, y) in enumerate(test_loader):
            res = model.forward(x)

            if verbose:
                print(f"x={x[0][0].squeeze():>5.0f} y={x[0][1].squeeze():>5.0f} " \
                    f"res={y.item():5.2f} pred={res.item():5.2f}", end="\r", flush=True)
            if idx == 50:
                break

    # save model
    save_model(model, operation_suffix)
    return model


def non_verbose_train(expr: str, operation_suffix: str, device: str) -> None:
    """
    Non verbose function to train the model with the specified parameters. Saves the model in every
    epoch specified in SAVE_EPOCHS. This function also do not evaluates any train/test accuracy due
    to maximize performance.

    Parameters:
        expr: the operation expression to train the model on
        operation_suffix: the suffix of the operation to create the save path with
        device: cpu or cuda device string

    Returns:
        model: trained pytorch model (but also saves the model in the specified path)
    """
    # input dimension of the model is the length of the dictionary
    model = MBAModel().to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"],
                           weight_decay=TRAIN_CONFIG["weight_decay"])
    train_loader, _ = get_loaders(expr, device)

    for _ in range(1, TRAIN_CONFIG["epochs"]+1):
        # train for one epoch
        loss_sum = 0.0
        for _, (X, y) in enumerate(train_loader):
            pred = model.forward(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        model.train()

    # save model
    save_model(model, operation_suffix)
    return model


def train_mapping(epochs: int, batch_size: int, dataset_size: int, device: str) -> None:
    """
    Function to train the network on the weights-operator mapping. Saves the model in every
    epoch specified in SAVE_EPOCHS. Prints the model status during the training.

    Parameters:
        epochs: the number of epochs to train the model
        batch_size: the batch size to use for training
        dataset_size: the size of the complete dataset
        device: cpu or cuda device string

    Returns:
        None
    """
    net = MappingModel().to(device)
    summary(net, input_size=(16768,), device=device)
    train_loader, test_loader = get_mapping_loaders(batch_size)
    try:
        next(iter(train_loader))
    except StopIteration:
        print("No data found in the training set.")
        return -1
    try:
        next(iter(test_loader))
    except StopIteration:
        print("No data found in the test set.")
        return -1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), 0.1, 0.9, 5e-4)

    for epoch in range(0, epochs):
        # every epoch a new progressbar is created
        # also, depending on the epoch the learning rate gets adjusted before
        # the network is set into training mode
        kbar = pkbar.Kbar(target=int(dataset_size/batch_size)-1, epoch=epoch, num_epochs=epochs,
                          width=20, always_stateful=True)
        adjust_learning_rate(optimizer, epoch, epochs, 0.1)
        net.train()
        correct = 0
        total = 0
        running_loss = 0.0

        # iterates over a batch of training data
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.squeeze().to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            _, predicted = outputs.max(1)

            # calculate the current running loss as well as the total accuracy
            # and update the progressbar accordingly
            running_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            kbar.update(batch_idx, values=[("loss", running_loss/(batch_idx+1)),
                                           ("acc", 100. * correct / total)])
        # calculate the test accuracy of the network at the end of each epoch
        with torch.no_grad():
            net.eval()
            t_total = 0
            t_correct = 0
            for _, (inputs_t, targets_t) in enumerate(test_loader):
                inputs_t = inputs_t.to(device)
                targets_t = targets_t.squeeze().to(device)
                outputs_t = net(inputs_t)
                _, predicted_t = outputs_t.max(1)
                t_total += targets_t.size(0)
                t_correct += predicted_t.eq(targets_t).sum().item()
            print(f"-> test acc: {100.*t_correct/t_total}")

    save_model(net, "mapping")

    # calculate the test accuracy of the network at the end of the training
    with torch.no_grad():
        net.eval()
        t_total = 0
        t_correct = 0
        for _, (inputs_t, targets_t) in enumerate(test_loader):
            inputs_t = inputs_t.to(device)
            targets_t = targets_t.squeeze().to(device)
            outputs_t = net(inputs_t)
            _, predicted_t = outputs_t.max(1)
            t_total += targets_t.size(0)
            t_correct += predicted_t.eq(targets_t).sum().item()


