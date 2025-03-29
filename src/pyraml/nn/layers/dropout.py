import numpy as np
import cupy as cp
from typing import Optional
from pyraml.core import Tensor
from pyraml.nn.module import Module
from pyraml.core.autograd import register_operation

class _DropoutNd(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace
        self.mask: Optional[Tensor] = None

    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'

class Dropout(_DropoutNd):
    @register_operation('dropout')
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        if x.device == 'cuda':
            self.mask = Tensor((cp.random.random(x.data.shape) > self.p) / (1 - self.p), 
                             device='cuda')
        else:
            self.mask = Tensor((np.random.random(x.data.shape) > self.p) / (1 - self.p), 
                             device='cpu')

        if self.inplace:
            x.data = x.data * self.mask.data
            return x
        return Tensor(x.data * self.mask.data, device=x.device)

class Dropout2d(_DropoutNd):
    @register_operation('dropout2d')
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        shape = x.data.shape
        if x.device == 'cuda':
            self.mask = Tensor(
                (cp.random.random((shape[0], shape[1], 1, 1)) > self.p) / (1 - self.p),
                device='cuda'
            )
        else:
            self.mask = Tensor(
                (np.random.random((shape[0], shape[1], 1, 1)) > self.p) / (1 - self.p),
                device='cpu'
            )

        if self.inplace:
            x.data = x.data * self.mask.data
            return x
        return Tensor(x.data * self.mask.data, device=x.device)

class Dropout3d(_DropoutNd):
    @register_operation('dropout3d')
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        shape = x.data.shape
        if x.device == 'cuda':
            self.mask = Tensor(
                (cp.random.random((shape[0], shape[1], 1, 1, 1)) > self.p) / (1 - self.p),
                device='cuda'
            )
        else:
            self.mask = Tensor(
                (np.random.random((shape[0], shape[1], 1, 1, 1)) > self.p) / (1 - self.p),
                device='cpu'
            )

        if self.inplace:
            x.data = x.data * self.mask.data
            return x
        return Tensor(x.data * self.mask.data, device=x.device)

class AlphaDropout(_DropoutNd):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p, inplace)
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    @register_operation('alpha_dropout')
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        if x.device == 'cuda':
            mask = (cp.random.random(x.data.shape) > self.p)
            noise = cp.random.normal(size=x.data.shape)
        else:
            mask = (np.random.random(x.data.shape) > self.p)
            noise = np.random.normal(size=x.data.shape)

        self.mask = Tensor(mask, device=x.device)
        
        a = (noise * self.alpha * (1 - mask)) / (1 - self.p)
        out = x.data * mask + a
        out = out * self.scale

        if self.inplace:
            x.data = out
            return x
        return Tensor(out, device=x.device)

@register_operation('dropout_backward')
def dropout_gradient(grad_output: Tensor, mask: Tensor) -> Tensor:
    return Tensor(grad_output.data * mask.data, device=grad_output.device)

@register_operation('dropout2d_backward')
def dropout2d_gradient(grad_output: Tensor, mask: Tensor) -> Tensor:
    return Tensor(grad_output.data * mask.data, device=grad_output.device)

@register_operation('dropout3d_backward')
def dropout3d_gradient(grad_output: Tensor, mask: Tensor) -> Tensor:
    return Tensor(grad_output.data * mask.data, device=grad_output.device)

@register_operation('alpha_dropout_backward')
def alpha_dropout_gradient(grad_output: Tensor, mask: Tensor, scale: float) -> Tensor:
    return Tensor(grad_output.data * mask.data * scale, device=grad_output.device)
