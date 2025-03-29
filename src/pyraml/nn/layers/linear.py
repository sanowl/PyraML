import numpy as np
from pyraml.core import Tensor
from pyraml.nn.layers.base import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Kaiming initialization
        self.weight = Tensor(
            np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weight.T
        if self.bias is not None:
            output += self.bias
        return output
