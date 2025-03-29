import numpy as np
from typing import Iterator, Optional
from pyraml.data.dataset import Dataset

class DataLoader:
    def __init__(
        self, 
        dataset: Dataset, 
        batch_size: int = 1, 
        shuffle: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield [self.dataset[j] for j in batch_indices]

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
