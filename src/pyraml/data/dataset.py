from abc import ABC, abstractmethod
from typing import Any, Tuple

class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
