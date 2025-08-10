import torch
import torch.nn as nn


def get_activation(value):
    if value == "relu":
        return nn.ReLU()
    elif value == "l_relu":
        # set the slope to align tensorflow
        return nn.LeakyReLU(negative_slope=0.2)
    elif value == "sigmoid":
        return nn.Sigmoid()
    elif value == "tanh":
        return nn.Tanh()
    elif value == "none":
        return DeactFunc()
    else:
        raise NotImplementedError("activation for the gating network not recognized")


class DeactFunc(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
