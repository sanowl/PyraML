from typing import Dict, Type, Optional
from pyraml.nn.layers.base import Module

class ModelRegistry:
    _models: Dict[str, Type[Module]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(model_cls: Type[Module]) -> Type[Module]:
            cls._models[name] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def get(cls, name: str) -> Optional[Type[Module]]:
        return cls._models.get(name)

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._models.keys())
