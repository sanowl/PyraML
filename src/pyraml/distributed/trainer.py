import ray
import numpy as np
from typing import Dict, Any, List, Optional
from pyraml.core import Tensor
from pyraml.nn.layers.base import Module

@ray.remote
class DistributedWorker:
    def __init__(self, model_cls, **kwargs):
        self.model = model_cls(**kwargs)
        self.optimizer = None
        self.local_batch_size = 0

    def train_step(self, batch_data, batch_target):
        outputs = self.model(batch_data)
        loss = self.criterion(outputs, batch_target)
        self.optimizer.zero_grad()
        loss.backward()
        return loss.data, self.get_gradients()

    def get_gradients(self) -> Dict[str, np.ndarray]:
        return {name: param.grad for name, param in self.model.named_parameters()}

    def update_parameters(self, parameters: Dict[str, np.ndarray]):
        for name, param in self.model.named_parameters():
            param.data = parameters[name]

class DistributedTrainer:
    def __init__(self, model_cls, num_workers: int = 4, strategy: str = "data_parallel"):
        ray.init()
        self.workers = [DistributedWorker.remote(model_cls) for _ in range(num_workers)]
        self.strategy = strategy
        self.parameter_server = self._setup_parameter_server()

    def _setup_parameter_server(self):
        if self.strategy == "data_parallel":
            return ray.remote(ParameterServer).remote()
        return None

    def train(self, dataset, epochs: int):
        for epoch in range(epochs):
            futures = []
            for worker, batch in zip(self.workers, self._distribute_data(dataset)):
                futures.append(worker.train_step.remote(*batch))
            
            losses, gradients = zip(*ray.get(futures))
            avg_gradients = self._average_gradients(gradients)
            self._update_all_workers(avg_gradients)
            
            avg_loss = np.mean(losses)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def _distribute_data(self, dataset):
        indices = np.array_split(np.arange(len(dataset)), len(self.workers))
        return [dataset[idx] for idx in indices]

    def _average_gradients(self, gradients: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        avg_grads = {}
        for key in gradients[0].keys():
            avg_grads[key] = np.mean([g[key] for g in gradients], axis=0)
        return avg_grads

    def _update_all_workers(self, gradients: Dict[str, np.ndarray]):
        update_futures = [w.update_parameters.remote(gradients) for w in self.workers]
        ray.get(update_futures)
