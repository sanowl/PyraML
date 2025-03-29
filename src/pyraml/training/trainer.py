from typing import Optional, Dict, Any, List, Callable, Union, Tuple
import numpy as np
from pyraml.core import Tensor
from pyraml.nn.module import Module
from pyraml.optim.optimizer import Optimizer
from pyraml.data.dataloader import DataLoader
from pyraml.core.context import no_grad
from pyraml.training.callbacks import Callback, EarlyStopping, ProgressBar

class Trainer:
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        device: str = 'cpu',
        metrics: Optional[Dict[str, Callable]] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.metrics = metrics or {}
        self.callbacks = callbacks or []
        self.history: Dict[str, List[float]] = {'loss': [], 'val_loss': []}
        self.best_state: Optional[Dict] = None
        self.best_metric: float = float('inf')
        self.stop_training = False
        self.epochs = 0
        self.train_loader = None
        
        # Initialize default callbacks
        if callbacks is None:
            callbacks = [ProgressBar()]
        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.trainer = self
        
    def train_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        metrics = {'loss': loss.item()}
        with no_grad():
            for name, metric_fn in self.metrics.items():
                metrics[name] = metric_fn(outputs, y)
                
        return metrics
    
    def validate_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        with no_grad():
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)
            
            metrics = {'val_loss': loss.item()}
            for name, metric_fn in self.metrics.items():
                metrics[f'val_{name}'] = metric_fn(outputs, y)
                
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        
        self.train_loader = train_loader
        self.epochs = epochs
        self.stop_training = False
        
        for epoch in range(epochs):
            if self.stop_training:
                break
                
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)
                
            self.model.train()
            train_metrics = []
            
            for batch in train_loader:
                metrics = self.train_step(batch)
                train_metrics.append(metrics)
            
            # Average training metrics
            epoch_metrics = {
                k: np.mean([m[k] for m in train_metrics]) 
                for k in train_metrics[0].keys()
            }
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_metrics = []
                
                for batch in val_loader:
                    metrics = self.validate_step(batch)
                    val_metrics.append(metrics)
                
                # Average validation metrics
                val_epoch_metrics = {
                    k: np.mean([m[k] for m in val_metrics])
                    for k in val_metrics[0].keys()
                }
                epoch_metrics.update(val_epoch_metrics)
            
            # Update history
            for k, v in epoch_metrics.items():
                self.history.setdefault(k, []).append(v)
            
            # Save best model if needed
            if save_best:
                current_metric = epoch_metrics.get(monitor, float('inf'))
                if (mode == 'min' and current_metric < self.best_metric) or \
                   (mode == 'max' and current_metric > self.best_metric):
                    self.best_metric = current_metric
                    self.best_state = self.model.state_dict()
            
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, epoch_metrics)
            
            # Print metrics
            metrics_str = ' - '.join(f'{k}: {v:.4f}' for k, v in epoch_metrics.items())
            print(f'Epoch {epoch+1}/{epochs} - {metrics_str}')
        
        return self.history
    
    def save_checkpoint(self, filepath: str) -> None:
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'best_state': self.best_state,
            'best_metric': self.best_metric
        }
        np.save(filepath, checkpoint)
    
    def load_checkpoint(self, filepath: str) -> None:
        checkpoint = np.load(filepath, allow_pickle=True).item()
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']
        self.best_state = checkpoint['best_state']
        self.best_metric = checkpoint['best_metric']
    
    def predict(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        x = x.to(self.device)
        
        self.model.eval()
        with no_grad():
            return self.model(x)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        test_metrics = []
        
        for batch in test_loader:
            metrics = self.validate_step(batch)
            test_metrics.append(metrics)
        
        return {
            k: np.mean([m[k] for m in test_metrics])
            for k in test_metrics[0].keys()
        }
