import numpy as np
import cupy as cp
from typing import Optional
from pyraml.core import Tensor
from pyraml.nn.module import Module
from pyraml.core.autograd import register_operation

class Activation(Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

class ReLU(Activation):
    @register_operation('relu')
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            x.data = np.maximum(0, x.data) if x.device == 'cpu' else cp.maximum(0, x.data)
            return x
        return Tensor(
            np.maximum(0, x.data) if x.device == 'cpu' else cp.maximum(0, x.data),
            device=x.device
        )

class LeakyReLU(Activation):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__(inplace)
        self.negative_slope = negative_slope

    @register_operation('leaky_relu')
    def forward(self, x: Tensor) -> Tensor:
        if x.device == 'cuda':
            result = cp.where(x.data > 0, x.data, self.negative_slope * x.data)
        else:
            result = np.where(x.data > 0, x.data, self.negative_slope * x.data)
        
        if self.inplace:
            x.data = result
            return x
        return Tensor(result, device=x.device)

class GELU(Activation):
    @register_operation('gelu')
    def forward(self, x: Tensor) -> Tensor:
        if x.device == 'cuda':
            return Tensor(
                x.data * 0.5 * (1.0 + cp.tanh(cp.sqrt(2.0 / cp.pi) * 
                (x.data + 0.044715 * cp.power(x.data, 3)))),
                device='cuda'
            )
        return Tensor(
            x.data * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * 
            (x.data + 0.044715 * np.power(x.data, 3)))),
            device='cpu'
        )

class Sigmoid(Activation):
    @register_operation('sigmoid')
    def forward(self, x: Tensor) -> Tensor:
        if x.device == 'cuda':
            result = 1 / (1 + cp.exp(-x.data))
        else:
            result = 1 / (1 + np.exp(-x.data))
            
        if self.inplace:
            x.data = result
            return x
        return Tensor(result, device=x.device)

class Tanh(Activation):
    @register_operation('tanh')
    def forward(self, x: Tensor) -> Tensor:
        if x.device == 'cuda':
            result = cp.tanh(x.data)
        else:
            result = np.tanh(x.data)
            
        if self.inplace:
            x.data = result
            return x
        return Tensor(result, device=x.device)

class SiLU(Activation):
    @register_operation('silu')
    def forward(self, x: Tensor) -> Tensor:
        if x.device == 'cuda':
            result = x.data * (1 / (1 + cp.exp(-x.data)))
        else:
            result = x.data * (1 / (1 + np.exp(-x.data)))
            
        if self.inplace:
            x.data = result
            return x
        return Tensor(result, device=x.device)

class Softmax(Activation):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    @register_operation('softmax')
    def forward(self, x: Tensor) -> Tensor:
        if x.device == 'cuda':
            max_val = cp.max(x.data, axis=self.dim, keepdims=True)
            exp_x = cp.exp(x.data - max_val)
            return Tensor(exp_x / cp.sum(exp_x, axis=self.dim, keepdims=True), device='cuda')
        else:
            max_val = np.max(x.data, axis=self.dim, keepdims=True)
            exp_x = np.exp(x.data - max_val)
            return Tensor(exp_x / np.sum(exp_x, axis=self.dim, keepdims=True), device='cpu')

# Register gradients for all activation functions
@register_operation('relu_backward')
def relu_gradient(grad_output: Tensor, x: Tensor) -> Tensor:
    if x.device == 'cuda':
        return Tensor(cp.where(x.data > 0, grad_output.data, 0), device='cuda')
    return Tensor(np.where(x.data > 0, grad_output.data, 0), device='cpu')

@register_operation('sigmoid_backward')
def sigmoid_gradient(grad_output: Tensor, x: Tensor) -> Tensor:
    sigmoid_x = 1 / (1 + (cp.exp(-x.data) if x.device == 'cuda' else np.exp(-x.data)))
    return Tensor(grad_output.data * sigmoid_x * (1 - sigmoid_x), device=x.device)

@register_operation('tanh_backward')
def tanh_gradient(grad_output: Tensor, x: Tensor) -> Tensor:
    tanh_x = cp.tanh(x.data) if x.device == 'cuda' else np.tanh(x.data)
    return Tensor(grad_output.data * (1 - tanh_x * tanh_x), device=x.device)
