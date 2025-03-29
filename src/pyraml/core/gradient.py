from typing import Dict, Set, List, Tuple, Optional, Callable, Any
import numpy as np
import weakref

class GradientTape:
    def __init__(self):
        self.operations: List[Tuple[str, List['Tensor'], 'Tensor']] = []
        self.gradient_functions: Dict[int, Callable] = {}
        self.requires_grad = True
        self._watch_set: Set[int] = set()
        
    def watch(self, tensor: 'Tensor') -> None:
        self._watch_set.add(id(tensor))
        
    def stop_watching(self, tensor: 'Tensor') -> None:
        self._watch_set.discard(id(tensor))
        
    def record_operation(self, op_name: str, inputs: List['Tensor'], output: 'Tensor', 
                        grad_fn: Callable) -> None:
        if not self.requires_grad:
            return
            
        self.operations.append((op_name, inputs, output))
        self.gradient_functions[id(output)] = grad_fn

class GradientAccumulator:
    def __init__(self):
        self._gradients: Dict[int, np.ndarray] = {}
        self._pending_grads: Dict[int, List[np.ndarray]] = {}
        
    def accumulate(self, tensor_id: int, gradient: np.ndarray) -> None:
        if tensor_id in self._pending_grads:
            self._pending_grads[tensor_id].append(gradient)
        else:
            self._pending_grads[tensor_id] = [gradient]
            
    def finalize(self) -> None:
        for tensor_id, grads in self._pending_grads.items():
            if len(grads) == 1:
                self._gradients[tensor_id] = grads[0]
            else:
                self._gradients[tensor_id] = sum(grads)
        self._pending_grads.clear()
        
    def get_gradient(self, tensor_id: int) -> Optional[np.ndarray]:
        return self._gradients.get(tensor_id)
        
    def clear(self) -> None:
        self._gradients.clear()
        self._pending_grads.clear()

class BackwardFunction:
    def __init__(self, op_name: str, saved_tensors: List['Tensor'], 
                 grad_fn: Callable):
        self.op_name = op_name
        self.saved_tensors = saved_tensors
        self.grad_fn = grad_fn
        
    def apply(self, grad_output: np.ndarray) -> List[Optional[np.ndarray]]:
        return self.grad_fn(grad_output, *[t.data for t in self.saved_tensors])

def register_gradient(op_name: str):
    def decorator(grad_fn: Callable):
        GradientRegistry.register(op_name, grad_fn)
        return grad_fn
    return decorator

class GradientRegistry:
    _gradients: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, op_name: str, grad_fn: Callable) -> None:
        cls._gradients[op_name] = grad_fn
        
    @classmethod
    def get(cls, op_name: str) -> Optional[Callable]:
        return cls._gradients.get(op_name)

@register_gradient('add')
def add_gradient(grad_output, x, y):
    return grad_output, grad_output

@register_gradient('mul')
def mul_gradient(grad_output, x, y):
    return grad_output * y, grad_output * x

@register_gradient('matmul')
def matmul_gradient(grad_output, x, y):
    return grad_output @ y.T, x.T @ grad_output

@register_gradient('exp')
def exp_gradient(grad_output, x):
    return grad_output * np.exp(x)

@register_gradient('log')
def log_gradient(grad_output, x):
    return grad_output / x
