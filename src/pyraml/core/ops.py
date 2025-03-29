import numpy as np
import numba
import cupy as cp
from typing import Union, Tuple, Optional, List, Dict
from .tensor import Tensor
from .autograd import register_operation

# Core CPU Operations with Numba optimization
@numba.jit(nopython=True, parallel=True, fastmath=True)
def _cpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b

@numba.jit(nopython=True, parallel=True)
def _cpu_elementwise_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

@numba.jit(nopython=True, parallel=True)
def _cpu_elementwise_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

@numba.jit(nopython=True)
def _cpu_conv2d_single(input_pad: np.ndarray, kernel: np.ndarray, 
                      out_h: int, out_w: int, stride_h: int, stride_w: int) -> np.ndarray:
    H_out, W_out = out_h, out_w
    K_h, K_w = kernel.shape
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride_h
            w_start = j * stride_w
            output[i, j] = np.sum(
                input_pad[h_start:h_start + K_h, w_start:w_start + K_w] * kernel
            )
    return output

class TensorOperations:
    # Basic Arithmetic Operations
    @staticmethod
    @register_operation('add')
    def add(a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
        if isinstance(b, (int, float)):
            return Tensor(a.data + b, device=a.device)
        if a.device != b.device:
            raise ValueError("Tensors must be on the same device")
        if a.device == 'cuda':
            return Tensor(cp.add(a.data, b.data), device='cuda')
        return Tensor(_cpu_elementwise_add(a.data, b.data))

    @staticmethod
    @register_operation('sub')
    def subtract(a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
        if isinstance(b, (int, float)):
            return Tensor(a.data - b, device=a.device)
        if a.device != b.device:
            raise ValueError("Tensors must be on the same device")
        if a.device == 'cuda':
            return Tensor(cp.subtract(a.data, b.data), device='cuda')
        return Tensor(a.data - b.data)

    @staticmethod
    @register_operation('mul')
    def multiply(a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
        if isinstance(b, (int, float)):
            return Tensor(a.data * b, device=a.device)
        if a.device != b.device:
            raise ValueError("Tensors must be on the same device")
        if a.device == 'cuda':
            return Tensor(cp.multiply(a.data, b.data), device='cuda')
        return Tensor(_cpu_elementwise_mul(a.data, b.data))

    @staticmethod
    @register_operation('div')
    def divide(a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
        if isinstance(b, (int, float)):
            return Tensor(a.data / b, device=a.device)
        if a.device != b.device:
            raise ValueError("Tensors must be on the same device")
        if a.device == 'cuda':
            return Tensor(cp.divide(a.data, b.data), device='cuda')
        return Tensor(a.data / b.data)

    # Matrix Operations
    @staticmethod
    @register_operation('matmul')
    def matmul(a: Tensor, b: Tensor) -> Tensor:
        if a.device != b.device:
            raise ValueError("Tensors must be on the same device")
        if a.device == 'cuda':
            return Tensor(cp.matmul(a.data, b.data), device='cuda')
        return Tensor(_cpu_matmul(a.data, b.data))

    @staticmethod
    @register_operation('transpose')
    def transpose(x: Tensor, dims: Optional[Tuple[int, ...]] = None) -> Tensor:
        if x.device == 'cuda':
            return Tensor(cp.transpose(x.data, dims), device='cuda')
        return Tensor(np.transpose(x.data, dims))

    # Reduction Operations
    @staticmethod
    @register_operation('sum')
    def sum(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        if x.device == 'cuda':
            return Tensor(cp.sum(x.data, axis=dim, keepdims=keepdim), device='cuda')
        return Tensor(np.sum(x.data, axis=dim, keepdims=keepdim))

    @staticmethod
    @register_operation('mean')
    def mean(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        if x.device == 'cuda':
            return Tensor(cp.mean(x.data, axis=dim, keepdims=keepdim), device='cuda')
        return Tensor(np.mean(x.data, axis=dim, keepdims=keepdim))

    @staticmethod
    @register_operation('max')
    def max(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        if x.device == 'cuda':
            return Tensor(cp.max(x.data, axis=dim, keepdims=keepdim), device='cuda')
        return Tensor(np.max(x.data, axis=dim, keepdims=keepdim))

    # Shape Operations
    @staticmethod
    def reshape(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
        if x.device == 'cuda':
            return Tensor(cp.reshape(x.data, shape), device='cuda')
        return Tensor(np.reshape(x.data, shape))

    @staticmethod
    def squeeze(x: Tensor, dim: Optional[int] = None) -> Tensor:
        if x.device == 'cuda':
            return Tensor(cp.squeeze(x.data, axis=dim), device='cuda')
        return Tensor(np.squeeze(x.data, axis=dim))

    @staticmethod
    def unsqueeze(x: Tensor, dim: int) -> Tensor:
        if x.device == 'cuda':
            return Tensor(cp.expand_dims(x.data, axis=dim), device='cuda')
        return Tensor(np.expand_dims(x.data, axis=dim))

    # Advanced Operations
    @staticmethod
    @register_operation('conv2d')
    def conv2d(x: Tensor, weight: Tensor, stride: Tuple[int, int] = (1, 1),
               padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1),
               groups: int = 1) -> Tensor:
        batch_size, in_channels, in_height, in_width = x.data.shape
        out_channels, _, kernel_height, kernel_width = weight.data.shape
        
        # Calculate output dimensions
        pad_height, pad_width = padding
        stride_height, stride_width = stride
        dilation_height, dilation_width = dilation
        
        out_height = ((in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) 
                     // stride_height + 1)
        out_width = ((in_width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1) 
                    // stride_width + 1)
        
        if x.device == 'cuda':
            # Use CuPy's implementation for GPU
            return Tensor(cp.conv2d(x.data, weight.data, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups), device='cuda')
        else:
            # CPU implementation with padding
            x_padded = np.pad(x.data, ((0, 0), (0, 0),
                                      (pad_height, pad_height),
                                      (pad_width, pad_width)))
            
            output = np.zeros((batch_size, out_channels, out_height, out_width))
            
            # Optimized convolution using Numba
            for b in range(batch_size):
                for c_out in range(out_channels):
                    for c_in in range(in_channels):
                        output[b, c_out] += _cpu_conv2d_single(
                            x_padded[b, c_in],
                            weight.data[c_out, c_in],
                            out_height, out_width,
                            stride_height, stride_width
                        )
            
            return Tensor(output)

    @staticmethod
    @register_operation('pool2d')
    def max_pool2d(x: Tensor, kernel_size: Tuple[int, int],
                   stride: Optional[Tuple[int, int]] = None,
                   padding: Tuple[int, int] = (0, 0)) -> Tensor:
        if stride is None:
            stride = kernel_size
            
        if x.device == 'cuda':
            return Tensor(cp.max_pool2d(x.data, kernel_size, stride, padding), device='cuda')
            
        batch_size, channels, height, width = x.data.shape
        k_h, k_w = kernel_size
        s_h, s_w = stride
        p_h, p_w = padding
        
        out_h = (height + 2 * p_h - k_h) // s_h + 1
        out_w = (width + 2 * p_w - k_w) // s_w + 1
        
        x_padded = np.pad(x.data, ((0, 0), (0, 0),
                                  (p_h, p_h), (p_w, p_w)),
                         mode='constant', constant_values=float('-inf'))
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s_h
                        w_start = j * s_w
                        output[b, c, i, j] = np.max(
                            x_padded[b, c,
                                   h_start:h_start + k_h,
                                   w_start:w_start + k_w]
                        )
        
        return Tensor(output)

    # Utility Operations
    @staticmethod
    def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
        if x.device == 'cuda':
            return Tensor(cp.where(condition.data, x.data, y.data), device='cuda')
        return Tensor(np.where(condition.data, x.data, y.data))

    @staticmethod
    def clip(x: Tensor, min_val: float, max_val: float) -> Tensor:
        if x.device == 'cuda':
            return Tensor(cp.clip(x.data, min_val, max_val), device='cuda')
        return Tensor(np.clip(x.data, min_val, max_val))

    @staticmethod
    def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
        device = tensors[0].device
        if device == 'cuda':
            return Tensor(cp.stack([t.data for t in tensors], axis=dim), device='cuda')
        return Tensor(np.stack([t.data for t in tensors], axis=dim))

    @staticmethod
    def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
        device = tensors[0].device
        if device == 'cuda':
            return Tensor(cp.concatenate([t.data for t in tensors], axis=dim), device='cuda')
        return Tensor(np.concatenate([t.data for t in tensors], axis=dim))
