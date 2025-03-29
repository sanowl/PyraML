from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pyraml.core import Tensor

class Optimizer(ABC):
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr
        self.state: Dict[str, Any] = {}

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None
