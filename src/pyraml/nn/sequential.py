from typing import Iterator, List, Optional, TypeVar, Union
from collections import OrderedDict
from pyraml.nn.module import Module
from pyraml.core import Tensor
import operator
from itertools import islice

T = TypeVar('T', bound=Module)

class Sequential(Module):
    def __init__(self, *args: Union[Module, OrderedDict[str, Module]]):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> T:
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx) -> Union[Module, 'Sequential']:
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, input: Tensor) -> Tensor:
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> 'Sequential':
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module) -> 'Sequential':
        n = len(self)
        if not (-n <= index <= n):
            raise IndexError('index {} is out of range'.format(index))
        if index < 0:
            index += n
        
        # Shift existing modules
        for i in range(n, index, -1):
            self.add_module(str(i), self._modules[str(i - 1)])
        self.add_module(str(index), module)
        return self

    def extend(self, modules: List[Module]) -> 'Sequential':
        for module in modules:
            self.append(module)
        return self

    def clear(self) -> None:
        self._modules.clear()
