from abc import ABC, abstractmethod
from typing import Dict, Any

class Module(ABC):
    def __init__(self):
        self.training = True
        self._parameters = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self) -> Dict[str, Any]:
        return self._parameters.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._parameters.update(state_dict)
