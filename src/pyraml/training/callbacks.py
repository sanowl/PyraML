from typing import Optional, Dict, Any
import numpy as np
from tqdm import tqdm

class Callback:
    def on_epoch_begin(self, epoch: int, logs: Dict[str, float] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        pass

class EarlyStopping(Callback):
    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0, 
                 patience: int = 0, mode: str = 'min', restore_best: bool = True):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best = restore_best
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.mode == 'min':
            improved = current < (self.best - self.min_delta)
        else:
            improved = current > (self.best + self.min_delta)

        if improved:
            self.best = current
            self.wait = 0
            if self.restore_best:
                self.best_weights = self.trainer.model.state_dict()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best and self.best_weights is not None:
                    self.trainer.model.load_state_dict(self.best_weights)
                self.trainer.stop_training = True

class ProgressBar(Callback):
    def __init__(self):
        self.epochs_pbar = None
        self.steps_pbar = None

    def on_epoch_begin(self, epoch: int, logs: Dict[str, float] = None) -> None:
        if self.epochs_pbar is None:
            self.epochs_pbar = tqdm(total=self.trainer.epochs, desc='Training')
        if self.steps_pbar is not None:
            self.steps_pbar.close()
        self.steps_pbar = tqdm(total=len(self.trainer.train_loader), 
                             desc=f'Epoch {epoch+1}', leave=False)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None) -> None:
        self.epochs_pbar.update(1)
        self.steps_pbar.close()
        metrics_str = ' - '.join(f'{k}: {v:.4f}' for k, v in logs.items())
        self.epochs_pbar.set_postfix_str(metrics_str)
