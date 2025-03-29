from typing import Dict, List, Optional, Union
import numpy as np
from pyraml.core import Tensor
from .optimizer import Optimizer, required

class SGD(Optimizer):
    def __init__(self, params, lr: float = required, momentum: float = 0,
                 dampening: float = 0, weight_decay: float = 0,
                 nesterov: bool = False, maximize: bool = False):
        if lr is required:
            raise ValueError("Learning rate required")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov,
                       maximize=maximize)
        
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                d_p_list.append(p.grad)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

            self._update_params(params_with_grad,
                              d_p_list,
                              momentum_buffer_list,
                              group)

        return loss

    def _update_params(self, params: List[Tensor],
                      d_p_list: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      group: Dict):
        
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        maximize = group['maximize']
        lr = group['lr']

        for i, param in enumerate(params):
            d_p = d_p_list[i]
            
            if maximize:
                d_p = -d_p

            if weight_decay != 0:
                d_p = d_p + weight_decay * param.data

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = np.copy(d_p)
                    momentum_buffer_list[i] = buf
                else:
                    buf = momentum * buf + (1 - dampening) * d_p
                    momentum_buffer_list[i] = buf

                if nesterov:
                    d_p = d_p + momentum * buf
                else:
                    d_p = buf

            param.data = param.data - lr * d_p

            # Update momentum_buffer in state
            if momentum != 0:
                state = self.state[param]
                state['momentum_buffer'] = momentum_buffer_list[i]
                
    def zero_momentum_buffers(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'momentum_buffer' in state:
                    del state['momentum_buffer']

    def set_momentum(self, momentum: float):
        for group in self.param_groups:
            group['momentum'] = momentum
            
    def get_momentum_buffers(self) -> Dict[int, np.ndarray]:
        buffers = {}
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'momentum_buffer' in state:
                    buffers[id(p)] = state['momentum_buffer']
        return buffers
