import numpy as np
import numba
import cupy as cp
from typing import Optional, Union, Tuple, List, Callable
from weakref import WeakKeyDictionary
from pyraml.core.gradient import GradientTape, GradientAccumulator, GradientRegistry
from .autograd import AutogradContext, register_operation

class Tensor:
    _cuda_enabled = hasattr(cp, '__version__')
    _memory_pool = WeakKeyDictionary()  # Smart memory management
    _current_tape: Optional[GradientTape] = None
    _gradient_accumulator = GradientAccumulator()
    
    def __init__(self, data: Union[np.ndarray, List, float], requires_grad: bool = False, 
                 device: str = 'cpu', memory_format: str = 'channels_last', _children=()): 
        self.device = device
        self.memory_format = memory_format
        self._init_data(data)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward_fn = lambda: None
        self._jit_cache = {}
        self._is_pinned = False
        self._prev = set(_children)
        self._grad_fn: Optional[Callable] = None
        
        if self.device == 'cpu' and self._is_pinned:
            self._pin_memory()
    
    def _init_data(self, data):
        if self.device == 'cuda' and self._cuda_enabled:
            self.data = cp.array(data, dtype=cp.float32)
            self._register_memory_pool()
            self._memory_pool[self] = cp.cuda.memory.MemoryPool().malloc
        else:
            self.data = np.array(data, dtype=np.float32)

    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _fast_matmul(a, b):
        return a @ b

    def to(self, device: str):
        if device == 'cuda' and not self._cuda_enabled:
            raise RuntimeError("CUDA not available")
        if device == self.device:
            return self
        
        if device == 'cuda':
            self.data = cp.array(self.data)
        else:
            self.data = cp.asnumpy(self.data)
        self.device = device
        return self

    def _register_memory_pool(self):
        if self.device == 'cuda':
            self._memory_pool[self] = cp.cuda.memory.MemoryPool()
    
    def _pin_memory(self):
        # Pin memory for faster GPU transfer
        self.data = np.ascontiguousarray(self.data)
    
    def to_gpu(self):
        if not self._cuda_enabled:
            raise RuntimeError("CUDA not available")
        if self.device == 'cuda':
            return self
        self.data = cp.array(self.data)
        self.device = 'cuda'
        return self
    
    def to_cpu(self):
        if self.device == 'cpu':
            return self
        self.data = cp.asnumpy(self.data)
        self.device = 'cpu'
        return self
    
    @register_operation('add')
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        def grad_left(grad): return grad
        def grad_right(grad): return grad
        return self._create_binary_operation(other, np.add, grad_left, grad_right)

    def add_(self, other: Union['Tensor', float]) -> 'Tensor':
        def grad_left(grad): return grad
        def grad_right(grad): return grad
        return self._create_binary_operation(other, np.add, grad_left, grad_right, inplace=True)

    @register_operation('mul')
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        def grad_left(grad): return grad * (other.data if isinstance(other, Tensor) else other)
        def grad_right(grad): return grad * self.data
        return self._create_binary_operation(other, np.multiply, grad_left, grad_right)

    def mul_(self, other: Union['Tensor', float]) -> 'Tensor':
        def grad_left(grad): return grad * (other.data if isinstance(other, Tensor) else other)
        def grad_right(grad): return grad * self.data
        return self._create_binary_operation(other, np.multiply, grad_left, grad_right, inplace=True)

    @register_operation('matmul')
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        result = Tensor(self._fast_matmul(self.data, other.data),
                       requires_grad=(self.requires_grad or other.requires_grad),
                       device=self.device)
        def grad_left(grad): return grad @ other.data.T
        def grad_right(grad): return self.data.T @ grad
        result._prev = {(self, grad_left), (other, grad_right)}
        return result

    @register_operation('exp')
    def exp(self) -> 'Tensor':
        result = Tensor(np.exp(self.data), requires_grad=self.requires_grad, device=self.device)
        if self.requires_grad:
            def grad_fn(grad): return grad * np.exp(self.data)
            result._prev = {(self, grad_fn)}
        return result

    @register_operation('log')
    def log(self) -> 'Tensor':
        result = Tensor(np.log(self.data), requires_grad=self.requires_grad, device=self.device)
        if self.requires_grad:
            def grad_fn(grad): return grad / self.data
            result._prev = {(self, grad_fn)}
        return result
    
    def mean(self) -> 'Tensor':
        return Tensor(self.data.mean())
    
    def item(self) -> float:
        return float(self.data)
    
    @property
    def T(self) -> 'Tensor':
        return Tensor(self.data.T)
    
    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data / other_data)
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims))
    
    def reshape(self, *shape: int) -> 'Tensor':
        return Tensor(self.data.reshape(*shape))
    
    def to_numpy(self) -> np.ndarray:
        return self.data

    def _check_device_consistency(self, other: 'Tensor') -> None:
        if self.device != other.device:
            raise RuntimeError(f"Expected both tensors to be on {self.device}, but found tensor on {other.device}")

    def zero_grad(self) -> None:
        if self.requires_grad:
            self._gradient_accumulator.clear()
            self.grad = None

    @staticmethod
    def _broadcast_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
        result = []
        for a, b in zip(reversed(shape1), reversed(shape2)):
            if a == 1 or b == 1:
                result.append(max(a, b))
            elif a == b:
                result.append(a)
            else:
                raise ValueError(f"Shapes {shape1} and {shape2} cannot be broadcast together")
        return tuple(reversed(result))

    def _handle_memory_format(self, other: 'Tensor') -> str:
        if self.memory_format == other.memory_format:
            return self.memory_format
        return 'channels_last'  # Default to channels_last if formats differ

    def clone(self) -> 'Tensor':
        return Tensor(
            self.data.copy(),
            requires_grad=self.requires_grad,
            device=self.device,
            memory_format=self.memory_format
        )

    def detach(self) -> 'Tensor':
        return Tensor(
            self.data,
            requires_grad=False,
            device=self.device,
            memory_format=self.memory_format
        )

    def backward(self, gradient: Optional[Union[np.ndarray, 'Tensor']] = None) -> None:
        if not self.requires_grad:
            return
        
        with AutogradContext.tape() as tape:
            if gradient is None:
                gradient = np.ones_like(self.data)
            elif isinstance(gradient, Tensor):
                gradient = gradient.data
                
            tape.watch(self)
            self._gradient_accumulator.accumulate(id(self), gradient)
            
            for op_name, inputs, output in reversed(tape.operations):
                if any(t.requires_grad for t in inputs):
                    grad_fn = GradientRegistry.get(op_name)
                    if grad_fn:
                        grads = grad_fn(self._gradient_accumulator.get_gradient(id(output)), 
                                      *[t.data for t in inputs])
                        for tensor, grad in zip(inputs, grads):
                            if tensor.requires_grad:
                                self._gradient_accumulator.accumulate(id(tensor), grad)
            
            self._gradient_accumulator.finalize()
            self.grad = self._gradient_accumulator.get_gradient(id(self))
