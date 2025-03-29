import numpy as np
from typing import Optional, Union, Tuple
from pyraml.core import Tensor
from pyraml.nn.module import Module, Parameter
from pyraml.nn import init

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: str = 'cpu', dtype: np.dtype = np.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Initialize weight with Kaiming/He initialization
        self.weight = Parameter(
            init.kaiming_uniform_(
                shape=(out_features, in_features),
                mode='fan_in',
                nonlinearity='relu'
            ),
            requires_grad=True,
            device=device
        )
        
        if bias:
            # Initialize bias with uniform bounds from He initialization
            bound = 1 / np.sqrt(in_features)
            self.bias = Parameter(
                init.uniform_(-bound, bound, shape=(out_features,)),
                requires_grad=True,
                device=device
            )
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight.data, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            bound = 1 / np.sqrt(self.in_features)
            init.uniform_(self.bias.data, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if input.device != self.device:
            input = input.to(self.device)
            
        # Reshape input if needed
        if len(input.data.shape) > 2:
            batch_size = input.data.shape[0]
            input_reshaped = input.reshape(batch_size, -1)
        else:
            input_reshaped = input

        # Perform matrix multiplication
        output = input_reshaped @ self.weight.T
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}')

    @property
    def weight_norm(self) -> float:
        return float(np.sqrt(np.sum(self.weight.data * self.weight.data)))

    def set_weight_norm(self, norm: float) -> None:
        if norm <= 0:
            raise ValueError(f"Weight norm must be positive, got {norm}")
        current_norm = self.weight_norm
        self.weight.data *= (norm / current_norm)

    def get_flops(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate FLOPs (multiply-adds) for this layer."""
        if len(input_shape) < 2:
            raise ValueError(f"Invalid input shape: {input_shape}")
        # Each output requires in_features multiplications and in_features-1 additions
        total_elements = np.prod(input_shape[:-1])  # Batch size and other dimensions
        return total_elements * self.out_features * (2 * self.in_features - 1)

    def get_memory_usage(self) -> int:
        """Calculate memory usage in bytes."""
        param_bytes = (self.weight.data.size + 
                      (self.bias.data.size if self.bias is not None else 0)) * 4  # float32
        # Add memory for gradients if requires_grad is True
        if self.weight.requires_grad:
            param_bytes *= 2  # Double for gradients
        return param_bytes
