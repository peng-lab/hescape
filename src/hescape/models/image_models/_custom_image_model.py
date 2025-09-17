from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hescape.models._utils import _crawl_and_import_custom_model


# (The CustomModel abstract class remains the same as before)
class CustomImageModel(nn.Module, ABC):
    """
    An abstract base class for defining a custom image model trunk.

    Users must inherit from this class and implement the abstract properties
    and methods to integrate their own models into the ImageEncoder.
    """

    def __init__(self, path: str):
        super().__init__()
        self.trunk = self.get_trunk()

        if not hasattr(self.trunk, "num_features"):
            raise AttributeError("The model returned by get_trunk() must have a 'num_features' attribute.")

        self.load_weights(path)

    @property
    @abstractmethod
    def total_blocks(self) -> int:
        pass

    @abstractmethod
    def get_trunk(self) -> nn.Module:
        pass

    def load_weights(self, path: str) -> None:
        if path and Path(path).exists():
            self.trunk.load_state_dict(torch.load(path, weights_only=True))
            print(f"Successfully loaded custom model weights from {path}")
        else:
            print(f"No weights found at {path}; using random initialization for custom model.")


def _build_custom_image_model(**kwargs: Any) -> tuple[nn.Module, int]:
    """Discovers and instantiates the user-defined subclass of CustomModel."""
    # 3. Dynamically load the user's code first
    custom_model_dir = _crawl_and_import_custom_model()
    weights_path = custom_model_dir / "pretrain_weights" / "custom_image_model"
    # 4. Now, discover the subclass which has been registered by the import
    subclasses = CustomImageModel.__subclasses__()

    if not subclasses:
        raise ImportError(
            "No custom model found. Ensure a class in the 'user_model' directory inherits from 'CustomModel'."
        )

    if len(subclasses) > 1:
        raise TypeError(
            f"Found multiple custom models: {[cls.__name__ for cls in subclasses]}. "
            "Only one subclass of 'CustomModel' is allowed in the 'user_model' directory."
        )

    custom_model_class = subclasses[0]
    print(f"Found custom model implementation: {custom_model_class.__name__}")

    instance = custom_model_class(weights_path)

    return instance.trunk, instance.total_blocks
