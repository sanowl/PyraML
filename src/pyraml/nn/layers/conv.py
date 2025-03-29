import numpy as np
from typing import Union, Tuple, Optional
from pyraml.core import Tensor
from pyraml.nn.module import Module, Parameter
from pyraml.nn import init
from pyraml.core.ops import TensorOperations

class _ConvNd(Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]], 
                 stride: Union[int, Tuple[int, ...]], 
                 padding: Union[int, Tuple[int, ...]], 
                 dilation: Union[int, Tuple[int, ...]], 
                 groups: int, bias: bool, 
                 padding_mode: str = 'zeros', 
                 device: str = 'cpu') -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.padding = padding if isinstance(padding, tuple) else (padding,)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device

        # Initialize weight using Kaiming/He initialization
        fan_in = in_channels * np.prod(self.kernel_size)
        self.weight = Parameter(
            init.kaiming_uniform_(
                shape=(out_channels, in_channels // groups, *self.kernel_size),
                mode='fan_in',
                nonlinearity='relu'
            ),
            requires_grad=True,
            device=device
        )

        if bias:
            bound = 1 / np.sqrt(fan_in)
            self.bias = Parameter(
                init.uniform_(-bound, bound, shape=(out_channels,)),
                requires_grad=True,
                device=device
            )
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight.data, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            fan_in = self.in_channels * np.prod(self.kernel_size)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias.data, -bound, bound)

    def extra_repr(self) -> str:
        return (f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, '
                f'stride={self.stride}, '
                f'padding={self.padding}, '
                f'dilation={self.dilation}, '
                f'groups={self.groups}, '
                f'bias={self.bias is not None}, '
                f'padding_mode={self.padding_mode}')

class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: str = 'cpu'
    ) -> None:
        kernel_size_tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride_tuple = (stride, stride) if isinstance(stride, int) else stride
        padding_tuple = (padding, padding) if isinstance(padding, int) else padding
        dilation_tuple = (dilation, dilation) if isinstance(dilation, int) else dilation

        super().__init__(
            in_channels, out_channels, kernel_size_tuple,
            stride_tuple, padding_tuple, dilation_tuple,
            groups, bias, padding_mode, device
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.device != self.device:
            input = input.to(self.device)

        return TensorOperations.conv2d(
            input, self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        ) + (self.bias.reshape(1, -1, 1, 1) if self.bias is not None else 0)

    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {input_shape}")
            
        batch_size, _, height, width = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        
        # Output dimensions
        out_height = ((height + 2 * pad_h - self.dilation[0] * (kernel_h - 1) - 1) 
                     // stride_h + 1)
        out_width = ((width + 2 * pad_w - self.dilation[1] * (kernel_w - 1) - 1) 
                    // stride_w + 1)
        
        # FLOPs per output element
        flops_per_element = (2 * kernel_h * kernel_w * self.in_channels // self.groups - 1)
        
        # Total FLOPs
        total_flops = (batch_size * self.out_channels * out_height * out_width * 
                      flops_per_element)
        
        if self.bias is not None:
            total_flops += batch_size * self.out_channels * out_height * out_width
            
        return total_flops

class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: str = 'cpu'
    ) -> None:
        kernel_size_tuple = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
        stride_tuple = (stride,) if isinstance(stride, int) else stride
        padding_tuple = (padding,) if isinstance(padding, int) else padding
        dilation_tuple = (dilation,) if isinstance(dilation, int) else dilation

        super().__init__(
            in_channels, out_channels, kernel_size_tuple,
            stride_tuple, padding_tuple, dilation_tuple,
            groups, bias, padding_mode, device
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.device != self.device:
            input = input.to(self.device)

        # Add dummy height dimension and use Conv2d implementation
        input_expanded = input.reshape(input.data.shape[0], input.data.shape[1], 1, -1)
        weight_expanded = self.weight.reshape(
            self.weight.data.shape[0], self.weight.data.shape[1], 1, -1
        )

        result = TensorOperations.conv2d(
            input_expanded, weight_expanded,
            stride=(1, self.stride[0]),
            padding=(0, self.padding[0]),
            dilation=(1, self.dilation[0]),
            groups=self.groups
        )

        if self.bias is not None:
            result = result + self.bias.reshape(1, -1, 1, 1)

        # Remove dummy height dimension
        return result.reshape(result.data.shape[0], result.data.shape[1], -1)
