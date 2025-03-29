import numpy as np
import cupy as cp
from typing import Optional, Tuple, Union
from pyraml.core import Tensor
from pyraml.core.autograd import register_operation

# Activation Functions
@register_operation('relu')
def relu(x: Tensor) -> Tensor:
    if x.device == 'cuda':
        return Tensor(cp.maximum(0, x.data), device='cuda')
    return Tensor(np.maximum(0, x.data), device=x.device)

@register_operation('gelu')
def gelu(x: Tensor) -> Tensor:
    if x.device == 'cuda':
        return Tensor(x.data * 0.5 * (1.0 + cp.tanh(cp.sqrt(2.0 / cp.pi) * (x.data + 0.044715 * cp.power(x.data, 3)))),
                     device='cuda')
    return Tensor(x.data * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * np.power(x.data, 3)))),
                 device=x.device)

@register_operation('softmax')
def softmax(x: Tensor, dim: int = -1) -> Tensor:
    if x.device == 'cuda':
        exp_x = cp.exp(x.data - cp.max(x.data, axis=dim, keepdims=True))
        return Tensor(exp_x / cp.sum(exp_x, axis=dim, keepdims=True), device='cuda')
    exp_x = np.exp(x.data - np.max(x.data, axis=dim, keepdims=True))
    return Tensor(exp_x / np.sum(exp_x, axis=dim, keepdims=True), device=x.device)

@register_operation('dropout')
def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0:
        return x
    
    if x.device == 'cuda':
        mask = (cp.random.random(x.data.shape) > p) / (1 - p)
        return Tensor(x.data * mask, device='cuda')
    mask = (np.random.random(x.data.shape) > p) / (1 - p)
    return Tensor(x.data * mask, device=x.device)

# Loss Functions
@register_operation('cross_entropy')
def cross_entropy(input: Tensor, target: Tensor, weight: Optional[Tensor] = None, 
                 reduction: str = 'mean') -> Tensor:
    log_probs = log_softmax(input, dim=1)
    if input.device == 'cuda':
        loss = -cp.sum(log_probs.data * target.data, dim=1)
    else:
        loss = -np.sum(log_probs.data * target.data, axis=1)
    
    if weight is not None:
        loss = loss * weight.data
        
    if reduction == 'mean':
        return Tensor(loss.mean(), device=input.device)
    elif reduction == 'sum':
        return Tensor(loss.sum(), device=input.device)
    return Tensor(loss, device=input.device)

@register_operation('log_softmax')
def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    if x.device == 'cuda':
        max_val = cp.max(x.data, axis=dim, keepdims=True)
        exp_x = cp.exp(x.data - max_val)
        log_sum_exp = cp.log(cp.sum(exp_x, axis=dim, keepdims=True))
        return Tensor(x.data - max_val - log_sum_exp, device='cuda')
    max_val = np.max(x.data, axis=dim, keepdims=True)
    exp_x = np.exp(x.data - max_val)
    log_sum_exp = np.log(np.sum(exp_x, axis=dim, keepdims=True))
    return Tensor(x.data - max_val - log_sum_exp, device=x.device)

# Normalization Functions
@register_operation('batch_norm')
def batch_norm(x: Tensor, running_mean: Optional[Tensor] = None, 
               running_var: Optional[Tensor] = None, training: bool = True, 
               momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    if training:
        if x.device == 'cuda':
            mean = cp.mean(x.data, axis=0)
            var = cp.var(x.data, axis=0)
        else:
            mean = np.mean(x.data, axis=0)
            var = np.var(x.data, axis=0)
            
        if running_mean is not None:
            running_mean.data = (1 - momentum) * running_mean.data + momentum * mean
        if running_var is not None:
            running_var.data = (1 - momentum) * running_var.data + momentum * var
    else:
        mean = running_mean.data if running_mean is not None else 0
        var = running_var.data if running_var is not None else 1
        
    return Tensor((x.data - mean) / np.sqrt(var + eps), device=x.device)

# Convolution Operations
@register_operation('conv2d')
def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
           dilation: Tuple[int, int] = (1, 1), groups: int = 1) -> Tensor:
    if x.device == 'cuda':
        result = cp.conv2d(x.data, weight.data, stride=stride, padding=padding,
                          dilation=dilation, groups=groups)
    else:
        result = _cpu_conv2d(x.data, weight.data, stride, padding, dilation, groups)
        
    if bias is not None:
        result += bias.data.reshape(1, -1, 1, 1)
    return Tensor(result, device=x.device)

# Pooling Operations
@register_operation('max_pool2d')
def max_pool2d(x: Tensor, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
        
    if x.device == 'cuda':
        return Tensor(cp.max_pool2d(x.data, kernel_size, stride, padding), device='cuda')
    return Tensor(_cpu_max_pool2d(x.data, kernel_size, stride, padding), device=x.device)

# Attention Mechanisms
@register_operation('scaled_dot_product_attention')
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, 
                               mask: Optional[Tensor] = None, dropout_p: float = 0.0) -> Tensor:
    d_k = query.data.shape[-1]
    scores = query @ key.T / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask.data
        
    attention = softmax(Tensor(scores), dim=-1)
    if dropout_p > 0:
        attention = dropout(attention, p=dropout_p)
        
    return attention @ value
