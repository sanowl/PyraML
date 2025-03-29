from typing import Dict, Iterator, Optional, Tuple, Union, List, Any
from collections import OrderedDict
import weakref
from pyraml.core import Tensor

class Parameter(Tensor):
    """A special kind of Tensor that represents a module parameter."""
    def __init__(self, data, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)
        self.grad_fn = None

class Module:
    _version = 1
    _parameters: Dict[str, Parameter]
    _buffers: Dict[str, Optional[Tensor]]
    _modules: Dict[str, 'Module']
    training: bool

    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        self.training = True
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"Expected Parameter but got {type(param)}")
        else:
            self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        if persistent:
            self._buffers[name] = tensor

    def add_module(self, name: str, module: Optional['Module']) -> None:
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"{name} is not a Module subclass")
        self._modules[name] = module

    def named_parameters(self, prefix: str = '') -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            if param is not None:
                yield prefix + ('.' if prefix else '') + name, param

        for name, module in self._modules.items():
            if module is not None:
                submodule_prefix = prefix + ('.' if prefix else '') + name
                yield from module.named_parameters(submodule_prefix)

    def parameters(self) -> Iterator[Parameter]:
        for _, param in self.named_parameters():
            yield param

    def train(self, mode: bool = True) -> 'Module':
        self.training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self

    def eval(self) -> 'Module':
        return self.train(False)

    def to(self, device: str) -> 'Module':
        for param in self.parameters():
            param.to(device)
        for buf in self._buffers.values():
            if buf is not None:
                buf.to(device)
        return self

    def cuda(self) -> 'Module':
        return self.to('cuda')

    def cpu(self) -> 'Module':
        return self.to('cpu')

    def zero_grad(self, set_to_none: bool = False) -> None:
        for param in self.parameters():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                state_dict[name] = param.data

        for name, buf in self._buffers.items():
            if buf is not None:
                state_dict[name] = buf.data

        for name, module in self._modules.items():
            if module is not None:
                state_dict[name] = module.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for name, param in self._parameters.items():
            if name in state_dict:
                param.data = state_dict[name]

        for name, buf in self._buffers.items():
            if name in state_dict and buf is not None:
                buf.data = state_dict[name]

        for name, module in self._modules.items():
            if name in state_dict and module is not None:
                module.load_state_dict(state_dict[name])

    def register_forward_hook(self, hook) -> 'RemovableHandle':
        handle = RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def register_forward_pre_hook(self, hook) -> 'RemovableHandle':
        handle = RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle

    def register_backward_hook(self, hook) -> 'RemovableHandle':
        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def __call__(self, *args, **kwargs):
        for hook in self._forward_pre_hooks.values():
            result = hook(self, args)
            if result is not None:
                args = result

        output = self.forward(*args, **kwargs)

        for hook in self._forward_hooks.values():
            hook_result = hook(self, args, output)
            if hook_result is not None:
                output = hook_result

        return output

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")

class RemovableHandle:
    next_id: int = 0

    def __init__(self, hooks_dict: Dict):
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __enter__(self) -> 'RemovableHandle':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
