from typing import List
from pyraml.core import Tensor
from pyraml.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        if momentum > 0:
            self.velocities = [None for _ in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            if self.momentum > 0:
                if self.velocities[i] is None:
                    self.velocities[i] = -self.lr * param.grad
                else:
                    self.velocities[i] = (self.momentum * self.velocities[i] - 
                                        self.lr * param.grad)
                param.data += self.velocities[i]
            else:
                param.data -= self.lr * param.grad
