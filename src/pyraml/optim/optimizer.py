from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator
from collections import defaultdict
import numpy as np
from pyraml.core import Tensor
from pyraml.nn.module import Parameter

class _RequiredParameter:
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class Optimizer(ABC):
    def __init__(self, params, defaults: Dict[str, Any]):
        self.defaults = defaults
        self._hook_for_profile = None
        self.state: Dict[Tensor, Dict[str, Any]] = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []

        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\nParameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dict."""
        state_dict = {
            'state': dict((str(k), v) for k, v in self.state.items()),
            'param_groups': self.param_groups,
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the optimizer state."""
        state = state_dict['state']
        groups = state_dict['param_groups']

        self.state = defaultdict(dict, state)
        self.param_groups = groups

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clears the gradients of all optimized parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.device == 'cuda':
                            p.grad.data.zero_()
                        else:
                            p.grad.data.fill(0)

    @abstractmethod
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        raise NotImplementedError

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Adds a parameter group to the optimizer's param_groups."""
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, '
                          'but the ordering of parameters in sets will change between runs.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                              "but one of the params is " + type(param).__name__)
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                              name)
            param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
