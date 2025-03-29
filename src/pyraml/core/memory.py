from typing import Dict, Any, Optional
import weakref

class MemoryManager:
    def __init__(self):
        self._allocation_cache: Dict[int, Any] = {}
        self._tensor_refs = weakref.WeakSet()
        
    def allocate(self, shape: tuple, dtype: str) -> Any:
        key = (shape, dtype)
        if key in self._allocation_cache:
            return self._allocation_cache[key].copy()
        return None
        
    def cache_allocation(self, tensor: 'Tensor') -> None:
        key = (tensor.data.shape, tensor.data.dtype)
        self._allocation_cache[key] = tensor.data
        self._tensor_refs.add(tensor)
        
    def clear_cache(self) -> None:
        self._allocation_cache.clear()
        
    def register_tensor(self, tensor: 'Tensor') -> None:
        self._tensor_refs.add(tensor)
        
    def get_memory_usage(self) -> int:
        return sum(t.data.nbytes for t in self._tensor_refs)

memory_manager = MemoryManager()
