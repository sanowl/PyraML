import numpy as np
from pyraml.core import Tensor
from pyraml.nn.layers.base import Module

class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target) ** 2).mean()

class CrossEntropyLoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        exp_pred = np.exp(pred.data)
        softmax_pred = exp_pred / exp_pred.sum(axis=1, keepdims=True)
        return Tensor(-np.log(softmax_pred[range(len(target)), target]))
