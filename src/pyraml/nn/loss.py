import numpy as np
from typing import Optional
from pyraml.core import Tensor
from pyraml.nn.module import Module
from pyraml.core.autograd import register_operation
from pyraml.core.functional import softmax, log_softmax

class Loss(Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def _reduce(self, loss: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class MSELoss(Loss):
    @register_operation('mse_loss')
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self._reduce((pred - target) ** 2)

class CrossEntropyLoss(Loss):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean', 
                 ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    @register_operation('cross_entropy_loss')
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.label_smoothing > 0:
            n_classes = pred.data.shape[-1]
            smooth_target = (1 - self.label_smoothing) * target.data
            smooth_target += self.label_smoothing / n_classes
            target = Tensor(smooth_target, device=target.device)

        log_probs = log_softmax(pred)
        loss = -log_probs * target

        if self.weight is not None:
            loss = loss * self.weight

        if self.ignore_index >= 0:
            mask = target.data != self.ignore_index
            loss = loss * Tensor(mask, device=loss.device)

        return self._reduce(loss)

class BCELoss(Loss):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean'):
        super().__init__(reduction)
        self.weight = weight

    @register_operation('bce_loss')
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-7
        pred = pred.clip(eps, 1 - eps)
        loss = -(target * pred.log() + (1 - target) * (1 - pred).log())
        
        if self.weight is not None:
            loss = loss * self.weight
            
        return self._reduce(loss)

class L1Loss(Loss):
    @register_operation('l1_loss')
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self._reduce(abs(pred - target))

class HuberLoss(Loss):
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.delta = delta

    @register_operation('huber_loss')
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        abs_diff = abs(diff)
        quadratic = 0.5 * diff ** 2
        linear = self.delta * abs_diff - 0.5 * self.delta ** 2
        loss = np.where(abs_diff <= self.delta, quadratic, linear)
        return self._reduce(Tensor(loss, device=pred.device))

class KLDivLoss(Loss):
    @register_operation('kl_div_loss')
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = target * (target.log() - pred)
        return self._reduce(loss)

# Register gradients for loss functions
@register_operation('mse_loss_backward')
def mse_loss_gradient(grad_output: Tensor, pred: Tensor, target: Tensor) -> Tensor:
    return grad_output * 2 * (pred - target)

@register_operation('bce_loss_backward')
def bce_loss_gradient(grad_output: Tensor, pred: Tensor, target: Tensor) -> Tensor:
    eps = 1e-7
    pred_clipped = pred.clip(eps, 1 - eps)
    return grad_output * (pred_clipped - target) / (pred_clipped * (1 - pred_clipped))

@register_operation('l1_loss_backward')
def l1_loss_gradient(grad_output: Tensor, pred: Tensor, target: Tensor) -> Tensor:
    return grad_output * np.sign(pred - target)

@register_operation('huber_loss_backward')
def huber_loss_gradient(grad_output: Tensor, pred: Tensor, target: Tensor, delta: float) -> Tensor:
    diff = pred - target
    abs_diff = abs(diff)
    grad = np.where(abs_diff <= delta, diff, delta * np.sign(diff))
    return grad_output * grad
