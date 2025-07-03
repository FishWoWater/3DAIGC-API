import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    PROCESSING = "processing"
    ERROR = "error"


class BaseModel(ABC):
    """Base class for all AI models"""

    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        feature_type: str = "unknown",
    ):
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.vram_requirement = vram_requirement  # MB
        self.feature_type = feature_type
        self.status = ModelStatus.UNLOADED
        self.gpu_id: Optional[int] = None
        self.model = None

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the actual model. Override in subclasses."""
        pass

    @abstractmethod
    def _unload_model(self) -> None:
        """Unload the actual model. Override in subclasses."""
        pass

    @abstractmethod
    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single request. Override in subclasses."""
        pass

    def load(self, gpu_id: int) -> bool:
        """Load model on specified GPU"""
        if self.status == ModelStatus.LOADED:
            return True

        try:
            self.status = ModelStatus.LOADING
            self.gpu_id = gpu_id

            # Set CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)

            # Load model
            self.model = self._load_model()
            self.status = ModelStatus.LOADED
            logger.info(f"Successfully loaded model {self.model_id} on GPU {gpu_id}")
            return True

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"Failed to load model {self.model_id}: {str(e)}")
            raise Exception(f"Failed to load model {self.model_id}: {str(e)}")

    def unload(self) -> bool:
        """Unload model from GPU"""
        if self.status == ModelStatus.UNLOADED:
            return True

        try:
            self._unload_model()
            self.model = None
            self.status = ModelStatus.UNLOADED
            self.gpu_id = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Successfully unloaded model {self.model_id}")
            return True

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"Failed to unload model {self.model_id}: {str(e)}")
            raise Exception(f"Failed to unload model {self.model_id}: {str(e)}")

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return results"""
        if self.status in [ModelStatus.UNLOADED, ModelStatus.LOADING]:
            raise Exception(
                f"Model {self.model_id} is not loaded, its status {self.status}"
            )

        try:
            self.status = ModelStatus.PROCESSING
            logger.info(f"Processing with model {self.model_id}")

            result = self._process_request(inputs)
            return result

        finally:
            # Reset status to loaded after processing
            self.status = ModelStatus.LOADED

    @abstractmethod
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_id": self.model_id,
            "feature_type": self.feature_type,
            "status": self.status.value,
            "gpu_id": self.gpu_id,
            "vram_requirement": self.vram_requirement,
            "supported_formats": self.get_supported_formats(),
        }
