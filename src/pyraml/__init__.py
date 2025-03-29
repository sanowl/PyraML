"""PyraML: A deep learning framework focusing on simplicity and educational purposes."""

__version__ = "0.1.0"

from pyraml.core import Tensor
from pyraml.nn.layers import Linear, Conv2d
from pyraml.nn.activations import ReLU, Sigmoid, Tanh
from pyraml.optim import SGD, Adam
from pyraml.nn.losses import MSELoss, CrossEntropyLoss
from pyraml.data import Dataset, DataLoader

__all__ = [
    "Tensor",
    "Linear", "Conv2d",
    "ReLU", "Sigmoid", "Tanh",
    "SGD", "Adam",
    "MSELoss", "CrossEntropyLoss",
    "Dataset", "DataLoader",
]
