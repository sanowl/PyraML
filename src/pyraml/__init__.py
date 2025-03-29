"""PyraML: A deep learning framework focusing on simplicity and educational purposes."""

# Import version, etc. but avoid importing implementation modules
__version__ = "0.1.0"

# Import core components
from pyraml.core import Tensor

# Import layers
from pyraml.nn.layers import (
    Linear, Conv2d,
    BatchNorm1d, BatchNorm2d, LayerNorm,
    Dropout, MaxPool2d
)

# Import activations
from pyraml.nn.activations import ReLU, Sigmoid, Tanh

# Import optimizers
from pyraml.optim import SGD, Adam

# Import losses
from pyraml.nn.loss import MSELoss, CrossEntropyLoss

# Import data utils
from pyraml.data import Dataset, DataLoader

__all__ = [
    "Tensor",
    "Linear", "Conv2d",
    "BatchNorm1d", "BatchNorm2d", "LayerNorm",
    "Dropout", "MaxPool2d",
    "ReLU", "Sigmoid", "Tanh",
    "SGD", "Adam",
    "MSELoss", "CrossEntropyLoss",
    "Dataset", "DataLoader",
]
