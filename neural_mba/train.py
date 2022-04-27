import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from typing import Any, Dict
from tqdm import trange
import numpy as np

from neural_mba.datasets import MBADataset
from neural_mba.models import MBAModel

SAVE_EPOCHS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
MODEL_PATH = "./models/"

TRAIN_CONFIG: Dict[str, Any] = {
    'training_expr' : '(x ^ y) + 2 * ( x & y)',
    'training_samples' : 10000,
    'device' : 'cpu',
    'epochs' : 3,
    'weight_decay' : 1e-4,
    'learning_rate': 0.001,
    'batch_size' : 8,
}
TEST_CONFIG: Dict[str, Any] = {
    'batch_size' : 1,
    'device' : 'cpu',
    'samples' : 100,
}


def save_model(net: nn.Sequential) -> None:
    """
    helper function which saves the given net in the specified path.
    if the path does not exists, it will be created.
    :param net: object of the model
    :return: None
    """
    print("\n[ Saving Model ]")
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    torch.save(state, MODEL_PATH+"mba_model")


def adjust_learning_rate(optimizer, epoch: int, epochs: int, learning_rate: int) -> None:
    """
    helper function to adjust the learning rate
    according to the current epoch to prevent overfitting.
    :paramo ptimizer: object of the used optimizer
    :param epoch: the current epoch
    :param epochs: the total epochs of the training
    :param learning_rate: the specified learning rate for the training
    :return: None
    """
    new_lr = learning_rate
    if epoch >= np.floor(epochs*0.5):
        new_lr /= 10
    if epoch >= np.floor(epochs*0.75):
        new_lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_loaders() -> DataLoader:
    """
    helper function to create dataset loaders
    :param batch_size: batch size which should be used for the dataloader
    :return: dataloader with the specified dataset
    """
    train_dataset = MBADataset(TRAIN_CONFIG['training_expr'], \
                     TRAIN_CONFIG['training_samples'], \
                     device=TRAIN_CONFIG['device'])
    train_loader = DataLoader(train_dataset,
                    batch_size=TRAIN_CONFIG['batch_size'], shuffle=True,
                    drop_last=True)
    test_dataset = MBADataset(TRAIN_CONFIG['training_expr'], \
                        TEST_CONFIG['samples'], \
                        device=TEST_CONFIG['device'])
    test_loader = DataLoader(test_dataset,
                    batch_size=TEST_CONFIG['batch_size'], shuffle=True,
                    drop_last=True)

    return train_loader, test_loader


def train() -> None:
    """
    Main function to train the model with the specified parameters. Saves the model in every
    epoch specified in SAVE_EPOCHS. Prints the model status during the training.
    :return: None
    """
    print("[ Initialize Training ]")

    # input dimension of the model is the length of the dictionary
    model = MBAModel().to(TRAIN_CONFIG['device'])
    summary(model, input_size=(2,))

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], \
                           weight_decay=TRAIN_CONFIG['weight_decay'])
    train_loader, test_loader = get_loaders()

    with trange(1, TRAIN_CONFIG['epochs']+1, bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
        losses = [0.0]
        accs = [0]

        for epoch in pbar:
            # train for one epoch
            loss_sum = 0.0
            for batch_idx, (X, y) in enumerate(train_loader):
                pbar.set_description(f"epoch {epoch:>3} batch {batch_idx:>3}/" \
                                     f"{(len(train_loader)//TRAIN_CONFIG['batch_size'])-1:>3}" \
                                     f"loss {loss_sum/max(TRAIN_CONFIG['batch_size']*batch_idx,1):10.5f}" \
                                     f" acc xx.xx")
                pred = model.forward(X)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            # evaluate performance
            pbar.set_description(f'epoch {epoch:>3} EVAL')
            model.eval()
            with torch.no_grad():
                no_correct = 0
                for test_batch_idx, (X, y) in enumerate(test_loader):
                    pred = model.forward(X)
                    # pred = torch.where(nn.Sigmoid()(pred) > 0.5, 1, 0)
                    # no_correct += (pred.squeeze() == y).type(torch.float).sum().item()
                    no_correct += torch.sum(torch.round(pred) == y).item()
                    pbar.set_description(f"epoch {epoch:>3} EVAL batch {test_batch_idx:>3}/" \
                                         f"{(len(test_loader)//TEST_CONFIG['batch_size'])-1:>3}")

            model.train()
            losses += [ loss_sum / len(train_loader) ]
            accs += [ no_correct / len(test_loader) * 100 ]

            pbar.set_description(f"epoch {epoch:>3} batch {batch_idx:>3}/" \
                                 f"{(len(train_loader)//TRAIN_CONFIG['batch_size'])-1:>3} " \
                                 f" loss {losses[-1]:10.5f} acc {accs[-1]:5.2f}")
            print()

        for idx, (x, y) in enumerate(test_loader):
            res = model.forward(x)
            print(f"x={x[0][0].squeeze():>5.0f} y={x[0][1].squeeze():>5.0f} " \
                  f"res={y.item():5.2f} pred={res.item():5.2f}", end="\r", flush=True)
            if idx == 50:
                break
    # save model
    save_model(model)
