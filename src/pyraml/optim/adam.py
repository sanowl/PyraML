from typing import Dict, List, Optional, Union
import numpy as np
from pyraml.core import Tensor
from .optimizer import Optimizer, required

class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0,
                 amsgrad: bool = False, maximize: bool = False,
                 foreach: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: Optional[bool] = None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       amsgrad=amsgrad, maximize=maximize, foreach=foreach,
                       capturable=capturable, differentiable=differentiable,
                       fused=fused)
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', False)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state.clear()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = np.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = np.zeros_like(p.data)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = np.zeros_like(p.data)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state['step'] += 1
                state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self._update_params(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group['amsgrad'],
                beta1,
                beta2,
                group['lr'],
                group['weight_decay'],
                group['eps'],
                group['maximize'],
                group['foreach'],
                group['capturable'],
                group['differentiable'],
                group['fused']
            )

        return loss

    def _update_params(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        maximize: bool,
        foreach: bool,
        capturable: bool,
        differentiable: bool,
        fused: Optional[bool]
    ) -> None:
        
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            if weight_decay != 0:
                grad = grad + weight_decay * param.data

            if maximize:
                grad = -grad

            # Decay the first and second moment running average coefficient
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * np.square(grad)

            if amsgrad:
                max_exp_avg_sqs[i] = np.maximum(max_exp_avg_sqs[i], exp_avg_sq)
                denom = np.sqrt(max_exp_avg_sqs[i]) + eps
            else:
                denom = np.sqrt(exp_avg_sq) + eps

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * np.sqrt(bias_correction2) / bias_correction1

            param.data = param.data - step_size * exp_avg / denom

            # Update state
            exp_avgs[i] = exp_avg
            exp_avg_sqs[i] = exp_avg_sq

    def get_momentum_buffers(self) -> Dict[int, np.ndarray]:
        buffers = {}
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'exp_avg' in state:
                    buffers[id(p)] = state['exp_avg']
        return buffers

    def load_state_dict(self, state_dict: Dict) -> None:
        super().load_state_dict(state_dict)
        # Upgrade from a previous version
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', False)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
