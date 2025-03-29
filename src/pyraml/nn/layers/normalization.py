import numpy as np
from typing import Optional, Tuple, Union
from pyraml.core import Tensor
from pyraml.nn.module import Module, Parameter
from pyraml.core.autograd import register_operation

class _NormBase(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device

        if self.affine:
            self.weight = Parameter(np.ones(num_features), requires_grad=True)
            self.bias = Parameter(np.zeros(num_features), requires_grad=True)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', np.zeros(num_features))
            self.register_buffer('running_var', np.ones(num_features))
            self.register_buffer('num_batches_tracked', 0)
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.fill(0)
            self.running_var.fill(1)
            self.num_batches_tracked = 0

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill(1)
            self.bias.data.fill(0)

class BatchNorm1d(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: str = 'cpu'
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device
        )

    @register_operation('batch_norm1d')
    def forward(self, x: Tensor) -> Tensor:
        if x.device != self.device:
            x = x.to(self.device)

        if self.training or not self.track_running_stats:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
                self.num_batches_tracked += 1
        else:
            mean = Tensor(self.running_mean, device=self.device)
            var = Tensor(self.running_var, device=self.device)

        x_norm = (x - mean) / (var + self.eps).sqrt()
        
        if self.affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm

class BatchNorm2d(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: str = 'cpu'
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device
        )

    @register_operation('batch_norm2d')
    def forward(self, x: Tensor) -> Tensor:
        if x.device != self.device:
            x = x.to(self.device)

        if self.training or not self.track_running_stats:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
                self.num_batches_tracked += 1
        else:
            mean = Tensor(self.running_mean, device=self.device)
            var = Tensor(self.running_var, device=self.device)

        x_norm = (x - mean.reshape(1, -1, 1, 1)) / (var.reshape(1, -1, 1, 1) + self.eps).sqrt()
        
        if self.affine:
            x_norm = x_norm * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)

        return x_norm

class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.device = device

        if self.elementwise_affine:
            self.weight = Parameter(np.ones(normalized_shape), requires_grad=True)
            self.bias = Parameter(np.zeros(normalized_shape), requires_grad=True)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            self.weight.data.fill(1)
            self.bias.data.fill(0)

    @register_operation('layer_norm')
    def forward(self, x: Tensor) -> Tensor:
        if x.device != self.device:
            x = x.to(self.device)

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()

        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm

# Register gradients
@register_operation('batch_norm_backward')
def batch_norm_gradient(grad_output: Tensor, x: Tensor, mean: Tensor, var: Tensor, 
                       weight: Optional[Tensor], eps: float) -> Tuple[Tensor, ...]:
    N = x.data.shape[0]
    std = (var + eps).sqrt()
    x_norm = (x - mean) / std
    
    if weight is not None:
        grad_output = grad_output * weight

    grad_input = grad_output / std
    grad_var = (-0.5 * grad_output * (x - mean) / (var + eps) ** 1.5).sum(0)
    grad_mean = (-grad_output / std).sum(0) + grad_var * (-2 * (x - mean)).mean(0)
    
    return grad_input + (grad_var * 2 * (x - mean) + grad_mean) / N

@register_operation('layer_norm_backward')
def layer_norm_gradient(grad_output: Tensor, x: Tensor, mean: Tensor, var: Tensor, 
                       weight: Optional[Tensor], eps: float) -> Tuple[Tensor, ...]:
    return batch_norm_gradient(grad_output, x, mean, var, weight, eps)
