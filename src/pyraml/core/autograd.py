from typing import Dict, List, Optional, Set, Tuple, Callable
import numpy as np
from contextlib import contextmanager
from .gradient import GradientTape, GradientRegistry
from .tensor import Tensor

class AutogradContext:
    _current_tape: Optional[GradientTape] = None
    _tape_stack: List[GradientTape] = []
    
    @classmethod
    @contextmanager
    def tape(cls):
        tape = GradientTape()
        cls._tape_stack.append(tape)
        cls._current_tape = tape
        try:
            yield tape
        finally:
            cls._tape_stack.pop()
            cls._current_tape = cls._tape_stack[-1] if cls._tape_stack else None
    
    @classmethod
    def record_operation(cls, op_name: str, inputs: List[Tensor], 
                        output: Tensor, grad_fn: Callable) -> None:
        if cls._current_tape is not None:
            cls._current_tape.record_operation(op_name, inputs, output, grad_fn)
            
    @classmethod
    def is_recording(cls) -> bool:
        return cls._current_tape is not None

def register_operation(op_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not AutogradContext.is_recording():
                return func(*args, **kwargs)
                
            inputs = [arg for arg in args if isinstance(arg, Tensor)]
            output = func(*args, **kwargs)
            grad_fn = GradientRegistry.get(op_name)
            
            if grad_fn and any(t.requires_grad for t in inputs):
                output.requires_grad = True
                AutogradContext.record_operation(op_name, inputs, output, grad_fn)
            
            return output
        return wrapper
    return decorator
