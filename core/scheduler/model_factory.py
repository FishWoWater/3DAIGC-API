"""
Model factory for dynamic model creation in multiprocessing environment.

This module provides utilities to create model instances from configuration
dictionaries, which is essential for worker processes that need to dynamically
load models without direct imports.
"""

import importlib
import logging
from typing import Any, Dict, Optional

from ..models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model instances from configuration"""

    # Registry of known adapter modules and classes
    ADAPTER_REGISTRY = {
        # TRELLIS adapters
        "trellis_text_to_textured_mesh": {
            "module": "adapters.trellis_adapter",
            "class": "TrellisTextToTexturedMeshAdapter",
        },
        "trellis_text_mesh_painting": {
            "module": "adapters.trellis_adapter",
            "class": "TrellisTextMeshPaintingAdapter",
        },
        "trellis_image_to_textured_mesh": {
            "module": "adapters.trellis_adapter",
            "class": "TrellisImageToTexturedMeshAdapter",
        },
        "trellis_image_mesh_painting": {
            "module": "adapters.trellis_adapter",
            "class": "TrellisImageMeshPaintingAdapter",
        },
        # Hunyuan3D adapters
        "hunyuan3d_2_0_image_to_raw_mesh": {
            "module": "adapters.hunyuan3d_adapter_v20",
            "class": "Hunyuan3DV20ImageToRawMeshAdapter",
        },
        "hunyuan3d_2_0_image_to_textured_mesh": {
            "module": "adapters.hunyuan3d_adapter_v20",
            "class": "Hunyuan3DV20ImageToTexturedMeshAdapter",
        },
        "hunyuan3d_2_0_image_mesh_painting": {
            "module": "adapters.hunyuan3d_adapter_v20",
            "class": "Hunyuan3DV20ImageMeshPaintingAdapter",
        },
        "hunyuan3d_image_to_raw_mesh": {
            "module": "adapters.hunyuan3d_adapter_v21",
            "class": "Hunyuan3DV21ImageToRawMeshAdapter",
        },
        "hunyuan3d_image_to_textured_mesh": {
            "module": "adapters.hunyuan3d_adapter_v21",
            "class": "Hunyuan3DV21ImageToTexturedMeshAdapter",
        },
        "hunyuan3d_image_mesh_painting": {
            "module": "adapters.hunyuan3d_adapter_v21",
            "class": "Hunyuan3DV21ImageMeshPaintingAdapter",
        },
        # PartField adapters
        "partfield_mesh_segmentation": {
            "module": "adapters.partfield_adapter",
            "class": "PartFieldSegmentationAdapter",
        },
        # HoloPart adapters
        "holopart_part_completion": {
            "module": "adapters.holopart_adapter",
            "class": "HoloPartCompletionAdapter",
        },
        # PartPacker adapters
        "partpacker_part_packing": {
            "module": "adapters.partpacker_adapter",
            "class": "PartPackerImageToRawMeshAdapter",
        },
        # UniRig adapters
        "unirig_auto_rig": {
            "module": "adapters.unirig_adapter",
            "class": "UniRigAdapter",
        },
    }

    @classmethod
    def register_adapter(cls, model_id: str, module_path: str, class_name: str):
        """Register a new adapter type"""
        cls.ADAPTER_REGISTRY[model_id] = {"module": module_path, "class": class_name}
        logger.info(f"Registered adapter: {model_id} -> {module_path}.{class_name}")

    @classmethod
    def create_model_from_config(cls, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance from configuration dictionary.

        Args:
            config: Model configuration containing:
                - model_id: Unique identifier for the model
                - module: Python module path (optional, can be inferred)
                - class: Model class name (optional, can be inferred)
                - init_params: Parameters for model initialization
                - feature_type: Type of feature this model handles
                - vram_requirement: VRAM requirement in MB

        Returns:
            BaseModel: Instantiated model object
        """
        try:
            model_id = config["model_id"]

            # Try to get module and class from registry first
            if model_id in cls.ADAPTER_REGISTRY:
                adapter_info = cls.ADAPTER_REGISTRY[model_id]
                module_name = adapter_info["module"]
                class_name = adapter_info["class"]
            elif "module" in config and "class" in config:
                # Use explicit module and class from config
                module_name = config["module"]
                class_name = config["class"]
            else:
                raise ValueError(
                    f"No adapter registration found for {model_id} and no explicit module/class provided"
                )

            logger.info(f"Creating model {model_id} from {module_name}.{class_name}")

            # Dynamic import
            try:
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
            except ImportError as e:
                logger.error(f"Failed to import {module_name}: {e}")
                raise ImportError(f"Cannot import module {module_name}: {e}")
            except AttributeError as e:
                logger.error(f"Class {class_name} not found in {module_name}: {e}")
                raise AttributeError(
                    f"Class {class_name} not found in module {module_name}: {e}"
                )

            # Create instance with config parameters
            init_params = config.get("init_params", {})

            # Add any additional parameters from config
            if "vram_requirement" in config:
                init_params["vram_requirement"] = config["vram_requirement"]

            model_instance = model_class(**init_params)

            # Verify the model has the expected properties
            if not hasattr(model_instance, "model_id"):
                model_instance.model_id = model_id
            if not hasattr(model_instance, "feature_type") and "feature_type" in config:
                model_instance.feature_type = config["feature_type"]

            logger.info(f"Successfully created model {model_id}")
            return model_instance

        except Exception as e:
            logger.error(f"Failed to create model from config: {e}")
            logger.error(f"Config: {config}")
            raise Exception(
                f"Failed to create model {config.get('model_id', 'unknown')}: {e}"
            )

    @classmethod
    def create_model_config(
        cls,
        model_id: str,
        feature_type: str,
        vram_requirement: int = 4096,
        max_workers: int = 1,
        init_params: Optional[Dict[str, Any]] = None,
        module_path: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a model configuration dictionary.

        Args:
            model_id: Unique identifier for the model
            feature_type: Type of feature this model handles
            vram_requirement: VRAM requirement in MB
            init_params: Parameters for model initialization
            module_path: Optional module path (inferred if not provided)
            class_name: Optional class name (inferred if not provided)

        Returns:
            Dict: Model configuration dictionary
        """
        config = {
            "model_id": model_id,
            "feature_type": feature_type,
            "vram_requirement": vram_requirement,
            "init_params": init_params or {},
        }

        # Add module and class if provided
        if module_path:
            config["module"] = module_path
        if class_name:
            config["class"] = class_name
        if max_workers:
            config["max_workers"] = max_workers

        return config

    @classmethod
    def get_available_adapters(cls) -> Dict[str, Dict[str, str]]:
        """Get list of available adapter types"""
        return cls.ADAPTER_REGISTRY.copy()

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        Validate a model configuration.

        Args:
            config: Model configuration to validate

        Returns:
            bool: True if valid, raises exception if invalid
        """
        required_fields = ["model_id", "feature_type"]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        model_id = config["model_id"]

        # Check if we can resolve the adapter
        if model_id not in cls.ADAPTER_REGISTRY:
            if "module" not in config or "class" not in config:
                raise ValueError(
                    f"Model {model_id} not in registry and no explicit module/class provided"
                )

        # Validate numeric fields
        if "vram_requirement" in config:
            if (
                not isinstance(config["vram_requirement"], int)
                or config["vram_requirement"] <= 0
            ):
                raise ValueError("vram_requirement must be a positive integer")

        return True


def create_model_from_config(config: Dict[str, Any]) -> BaseModel:
    """
    Convenience function to create a model from configuration.
    This is the function used by worker processes.
    """
    return ModelFactory.create_model_from_config(config)


def get_model_configs_from_settings(
    models_config: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Generate model configurations from already-parsed settings models configuration.

    Args:
        models_config: Parsed models configuration from settings (feature_type -> model_id -> ModelConfig or dict)

    Returns:
        Dict: Mapping of model_id to configuration for enabled models only
    """
    configs = {}

    # Process each feature type and its models
    for feature_type, models in models_config.items():
        if not isinstance(models, dict):
            continue

        for model_id, model_config in models.items():
            # # Determine if it's a ModelConfig object or dict
            is_model_config_obj = hasattr(model_config, "vram_requirement")

            # Skip disabled models - handle both ModelConfig objects and dicts
            if is_model_config_obj:
                # It's a ModelConfig object
                enabled = getattr(model_config, "enabled", True)
                if not enabled:
                    logger.info(f"Skipping disabled model: {model_id}")
                    continue

                vram_requirement = getattr(model_config, "vram_requirement", 4096)
                model_path = getattr(model_config, "model_path", None)
                supported_inputs = getattr(model_config, "supported_inputs", [])
                supported_outputs = getattr(model_config, "supported_outputs", [])
                max_workers = getattr(model_config, "max_workers", 1)
            else:
                logger.warning(f"Unknown model config type for {model_id}, skipping")
                continue

            # Check if model_id is in our adapter registry
            if model_id not in ModelFactory.ADAPTER_REGISTRY:
                logger.warning(
                    f"Model {model_id} not found in adapter registry, skipping"
                )
                continue

            # Create configuration dictionary
            config = ModelFactory.create_model_config(
                model_id=model_id,
                feature_type=feature_type,
                vram_requirement=vram_requirement,
                max_workers=max_workers,
                init_params={},
            )

            # Add additional configuration
            if model_path:
                config["model_path"] = model_path
            if supported_inputs:
                config["supported_inputs"] = supported_inputs
            if supported_outputs:
                config["supported_outputs"] = supported_outputs

            configs[model_id] = config
            logger.debug(f"Configured model {model_id} for feature {feature_type}")

    logger.info(f"Configured {len(configs)} enabled models from settings")
    return configs


def get_default_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Fallback function to generate default model configurations when YAML loading fails.

    Returns:
        Dict: Mapping of model_id to configuration
    """
    configs = {}

    # TRELLIS models
    configs.update(
        {
            "trellis_text_to_textured_mesh": ModelFactory.create_model_config(
                model_id="trellis_text_to_textured_mesh",
                feature_type="text_to_textured_mesh",
                vram_requirement=11776,  # 12GB
            ),
            "trellis_text_mesh_painting": ModelFactory.create_model_config(
                model_id="trellis_text_mesh_painting",
                feature_type="text_mesh_painting",
                vram_requirement=11776,
            ),
            "trellis_image_to_textured_mesh": ModelFactory.create_model_config(
                model_id="trellis_image_to_textured_mesh",
                feature_type="image_to_textured_mesh",
                vram_requirement=11776,
            ),
            "trellis_image_mesh_painting": ModelFactory.create_model_config(
                model_id="trellis_image_mesh_painting",
                feature_type="image_mesh_painting",
                vram_requirement=11776,
            ),
        }
    )

    # Hunyuan3D2.0 models
    configs.update(
        {
            "hunyuan3d_2_0_image_to_raw_mesh": ModelFactory.create_model_config(
                model_id="hunyuan3d_2_0_image_to_raw_mesh",
                feature_type="image_to_raw_mesh",
                vram_requirement=5120,
            ),
            "hunyuan3d_2_0_image_to_textured_mesh": ModelFactory.create_model_config(
                model_id="hunyuan3d_2_0_image_to_textured_mesh",
                feature_type="image_to_textured_mesh",
                vram_requirement=14336,
            ),
            "hunyuan3d_2_0_image_mesh_painting": ModelFactory.create_model_config(
                model_id="hunyuan3d_2_0_image_mesh_painting",
                feature_type="image_mesh_painting",
                vram_requirement=11264,
            ),
        }
    )

    # Hunyuan3D2.1 models
    configs.update(
        {
            "hunyuan3d_image_to_raw_mesh": ModelFactory.create_model_config(
                model_id="hunyuan3d_image_to_raw_mesh",
                feature_type="image_to_raw_mesh",
                vram_requirement=8192,
            ),
            "hunyuan3d_image_to_textured_mesh": ModelFactory.create_model_config(
                model_id="hunyuan3d_image_to_textured_mesh",
                feature_type="image_to_textured_mesh",
                vram_requirement=19456,
            ),
            "hunyuan3d_image_mesh_painting": ModelFactory.create_model_config(
                model_id="hunyuan3d_image_mesh_painting",
                feature_type="image_mesh_painting",
                vram_requirement=12288,
            ),
        }
    )

    # PartField models
    configs.update(
        {
            "partfield_mesh_segmentation": ModelFactory.create_model_config(
                model_id="partfield_mesh_segmentation",
                feature_type="mesh_segmentation",
                vram_requirement=4096,  # 4GB
            )
        }
    )

    # HoloPart models
    configs.update(
        {
            "holopart_part_completion": ModelFactory.create_model_config(
                model_id="holopart_part_completion",
                feature_type="part_completion",
                vram_requirement=10240,  # 10GB
            )
        }
    )

    # PartPacker models
    configs.update(
        {
            "partpacker_part_packing": ModelFactory.create_model_config(
                model_id="partpacker_part_packing",
                feature_type="part_packing",
                vram_requirement=10240,  # 10GB
            )
        }
    )

    # UniRig models
    configs.update(
        {
            "unirig_auto_rig": ModelFactory.create_model_config(
                model_id="unirig_auto_rig",
                feature_type="auto_rig",
                vram_requirement=9216,  # 9GB
            )
        }
    )

    return configs
