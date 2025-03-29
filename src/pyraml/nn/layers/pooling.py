import numpy as np
from typing import Union, Tuple, Optional
from pyraml.core import Tensor
from pyraml.nn.module import Module
from pyraml.core.autograd import register_operation

class _PoolNd(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False
    ):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride,)
        self.padding = padding if isinstance(padding, tuple) else (padding,)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return (f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, dilation={self.dilation}, '
                f'ceil_mode={self.ceil_mode}')

    def _compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        dims = []
        for i, (input_dim, kernel, pad, stride, dilation) in enumerate(
            zip(input_shape[2:], self.kernel_size, self.padding, 
                self.stride, self.dilation)
        ):
            if self.ceil_mode:
                dim = ((input_dim + 2 * pad - dilation * (kernel - 1) - 1 + stride) 
                      // stride)
            else:
                dim = ((input_dim + 2 * pad - dilation * (kernel - 1) - 1) 
                      // stride + 1)
            dims.append(dim)
        return (input_shape[0], input_shape[1], *dims)

class MaxPool2d(_PoolNd):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False
    ):
        super().__init__(
            kernel_size, stride, padding,
            dilation, return_indices, ceil_mode
        )
        self.indices = None

    @register_operation('maxpool2d')
    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        output_shape = self._compute_output_shape(x.data.shape)
        batch_size, channels, height, width = x.data.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # Pad input
        if any(p > 0 for p in self.padding):
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
                mode='constant',
                constant_values=float('-inf')
            )
        else:
            x_padded = x.data
        
        output = np.zeros(output_shape)
        self.indices = np.zeros_like(output, dtype=np.int64)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_shape[2]):
                    for j in range(output_shape[3]):
                        h_start = i * s_h
                        w_start = j * s_w
                        h_end = min(h_start + k_h, height + 2 * p_h)
                        w_end = min(w_start + k_w, width + 2 * p_w)
                        
                        pool_region = x_padded[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.max(pool_region)
                        self.indices[b, c, i, j] = np.argmax(pool_region.reshape(-1))
        
        result = Tensor(output, device=x.device)
        return (result, Tensor(self.indices)) if self.return_indices else result

class AvgPool2d(_PoolNd):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True
    ):
        super().__init__(kernel_size, stride, padding, 1, False, ceil_mode)
        self.count_include_pad = count_include_pad

    @register_operation('avgpool2d')
    def forward(self, x: Tensor) -> Tensor:
        output_shape = self._compute_output_shape(x.data.shape)
        batch_size, channels, height, width = x.data.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # Pad input
        if any(p > 0 for p in self.padding):
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
                mode='constant',
                constant_values=0
            )
        else:
            x_padded = x.data
        
        output = np.zeros(output_shape)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_shape[2]):
                    for j in range(output_shape[3]):
                        h_start = i * s_h
                        w_start = j * s_w
                        h_end = min(h_start + k_h, height + 2 * p_h)
                        w_end = min(w_start + k_w, width + 2 * p_w)
                        
                        pool_region = x_padded[b, c, h_start:h_end, w_start:w_end]
                        if self.count_include_pad:
                            divisor = k_h * k_w
                        else:
                            divisor = np.sum(pool_region != 0)
                        output[b, c, i, j] = np.sum(pool_region) / divisor
        
        return Tensor(output, device=x.device)

# Register gradients
@register_operation('maxpool2d_backward')
def maxpool2d_gradient(grad_output: Tensor, input: Tensor, indices: Tensor,
                      kernel_size: Tuple[int, int], stride: Tuple[int, int],
                      padding: Tuple[int, int]) -> Tensor:
    grad_input = np.zeros_like(input.data)
    batch_size, channels, height, width = input.data.shape
    
    for b in range(batch_size):
        for c in range(channels):
            for i in range(grad_output.data.shape[2]):
                for j in range(grad_output.data.shape[3]):
                    h_start = i * stride[0]
                    w_start = j * stride[1]
                    idx = indices.data[b, c, i, j]
                    h_offset = idx // kernel_size[1]
                    w_offset = idx % kernel_size[1]
                    grad_input[b, c, h_start + h_offset, w_start + w_offset] += \
                        grad_output.data[b, c, i, j]
    
    return Tensor(grad_input)

@register_operation('avgpool2d_backward')
def avgpool2d_gradient(grad_output: Tensor, input: Tensor,
                      kernel_size: Tuple[int, int], stride: Tuple[int, int],
                      padding: Tuple[int, int], count_include_pad: bool) -> Tensor:
    grad_input = np.zeros_like(input.data)
    batch_size, channels, height, width = input.data.shape
    
    for b in range(batch_size):
        for c in range(channels):
            for i in range(grad_output.data.shape[2]):
                for j in range(grad_output.data.shape[3]):
                    h_start = i * stride[0]
                    w_start = j * stride[1]
                    h_end = min(h_start + kernel_size[0], height)
                    w_end = min(w_start + kernel_size[1], width)
                    
                    if count_include_pad:
                        divisor = kernel_size[0] * kernel_size[1]
                    else:
                        divisor = (h_end - h_start) * (w_end - w_start)
                    
                    grad_input[b, c, h_start:h_end, w_start:w_end] += \
                        grad_output.data[b, c, i, j] / divisor
    
    return Tensor(grad_input)
