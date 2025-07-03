"""
UniRig Inference Engine - Optimized class-based inference system

This module provides a fast, class-based alternative to the original PyTorch Lightning
trainer-based inference system. It loads models once and reuses them for multiple
inference runs, eliminating the overhead of repeated checkpoint loading.
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from box import Box

from thirdparty.UniRig.src.data.datapath import Datapath
from thirdparty.UniRig.src.data.dataset import DatasetConfig, UniRigDatasetModule
from thirdparty.UniRig.src.data.extract import extract_builtin
from thirdparty.UniRig.src.data.transform import TransformConfig
from thirdparty.UniRig.src.inference.download import download
from thirdparty.UniRig.src.model.parse import get_model
from thirdparty.UniRig.src.system.parse import get_system, get_writer
from thirdparty.UniRig.src.tokenizer.parse import get_tokenizer
from thirdparty.UniRig.src.tokenizer.spec import TokenizerConfig

CONFIG_PREFIX = "thirdparty/UniRig"


@dataclass
class InferenceConfig:
    """Configuration for inference engine"""

    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "bf16-mixed"  # bf16-mixed, fp16, fp32
    compile_model: bool = False  # Use torch.compile for faster inference
    cache_dir: str = "tmp"
    num_workers: int = 4
    batch_size: int = 1


@dataclass
class LoadedSystem:
    """Container for a fully loaded and ready-to-use system"""

    system: torch.nn.Module
    model: torch.nn.Module
    tokenizer_config: Optional[Dict]
    data_config: Box
    transform_config: Box
    writer_config: Dict
    task_config: Box

    def __post_init__(self):
        """Ensure system is in eval mode"""
        self.system.eval()


class ModelCache:
    """Cache for loaded models and systems to avoid repeated loading"""

    def __init__(self, device: torch.device):
        self._loaded_systems = {}  # Cache fully loaded systems with checkpoints
        self._models = {}
        self._tokenizers = {}
        self._configs = {}  # Cache parsed configurations
        self.device = device

    def get_loaded_system(
        self, task_config_path: str, compile_model: bool = True
    ) -> LoadedSystem:
        """Get a fully loaded system with checkpoint, configs, and everything ready"""
        if task_config_path not in self._loaded_systems:
            self._loaded_systems[task_config_path] = self._load_complete_system(
                task_config_path, compile_model
            )

        return self._loaded_systems[task_config_path]

    def _load_complete_system(
        self, task_config_path: str, compile_model: bool
    ) -> LoadedSystem:
        """Load and prepare a complete system with all configurations"""
        # Load task configuration
        task_config = self._load_config(task_config_path)

        # Load all sub-configurations
        data_config = self._load_config(
            f"{CONFIG_PREFIX}/configs/data/{task_config.components.data}"
        )
        transform_config = self._load_config(
            f"{CONFIG_PREFIX}/configs/transform/{task_config.components.transform}"
        )

        # Get tokenizer config if needed
        tokenizer_config = None
        if task_config.components.get("tokenizer"):
            tokenizer_config = self._load_config(
                f"{CONFIG_PREFIX}/configs/tokenizer/{task_config.components.tokenizer}"
            )

        # Get and cache model
        model_config = self._load_config(
            f"{CONFIG_PREFIX}/configs/model/{task_config.components.model}"
        )
        model_key = f"{task_config.components.model}_{task_config.components.get('tokenizer', 'no_tokenizer')}"
        model = self._get_model(model_key, model_config, tokenizer_config)

        # Get and cache system
        system_config = self._load_config(
            f"{CONFIG_PREFIX}/configs/system/{task_config.components.system}"
        )
        system_key = f"{task_config.components.system}_{model_key}"
        system = self._get_system(system_key, system_config, model)

        # Load checkpoint (this is the expensive operation we want to do once)
        checkpoint_path = self._get_checkpoint_path(task_config.resume_from_checkpoint)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )  # weights_only=True will fail
        system.load_state_dict(checkpoint["state_dict"])
        system.to(self.device)
        system.eval()

        # Compile model for faster inference
        if compile_model:
            system = torch.compile(system)

        # Prepare writer config template
        writer_config = task_config.writer.copy()

        return LoadedSystem(
            system=system,
            model=model,
            tokenizer_config=tokenizer_config,
            data_config=data_config,
            transform_config=transform_config,
            writer_config=writer_config,
            task_config=task_config,
        )

    def _get_model(
        self,
        model_key: str,
        model_config: Dict,
        tokenizer_config: Optional[Dict] = None,
    ):
        """Get cached model or load new one"""
        if model_key not in self._models:
            tokenizer = None
            if tokenizer_config is not None:
                tokenizer_key = str(tokenizer_config)
                if tokenizer_key not in self._tokenizers:
                    tokenizer_config_obj = TokenizerConfig.parse(
                        config=tokenizer_config
                    )
                    self._tokenizers[tokenizer_key] = get_tokenizer(
                        config=tokenizer_config_obj
                    )
                tokenizer = self._tokenizers[tokenizer_key]

            self._models[model_key] = get_model(tokenizer=tokenizer, **model_config)

        return self._models[model_key]

    def _get_system(
        self, system_key: str, system_config: Dict, model, steps_per_epoch: int = 1
    ):
        """Get cached system or create new one"""
        if system_key not in self._models:  # Using _models cache for systems too
            self._models[system_key] = get_system(
                **system_config,
                model=model,
                steps_per_epoch=steps_per_epoch,
            )

        return self._models[system_key]

    def _load_config(self, config_path: str) -> Box:
        """Load and cache YAML configuration file"""
        if config_path not in self._configs:
            if not config_path.endswith(".yaml"):
                config_path += ".yaml"

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, "r") as f:
                self._configs[config_path] = Box(yaml.safe_load(f))

        return self._configs[config_path]

    def _get_checkpoint_path(self, checkpoint_name: str) -> str:
        """Get checkpoint path, downloading if necessary (cached in parent class)"""
        # This will be handled by the parent class's checkpoint cache
        return download(checkpoint_name)

    def clear(self):
        """Clear all cached models and systems"""
        self._loaded_systems.clear()
        self._models.clear()
        self._tokenizers.clear()
        self._configs.clear()


class UniRigInferenceEngine:
    """
    Fast inference engine for UniRig automatic rigging

    This class provides an optimized inference interface that:
    - Loads models once and reuses them
    - Supports batch processing
    - Uses torch.compile for faster inference
    - Provides simple API for both skeleton and skin generation
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        self.model_cache = ModelCache(self.device)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Set global precision and seed
        self._setup_environment()

        # Loaded checkpoint paths to avoid re-downloading
        self._checkpoint_cache = {}

        # Default task configurations
        self.default_skeleton_config = f"{CONFIG_PREFIX}/configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
        self.default_skin_config = (
            f"{CONFIG_PREFIX}/configs/task/quick_inference_unirig_skin.yaml"
        )

    def _setup_environment(self):
        """Set up PyTorch environment for optimal inference"""
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _load_config(self, config_path: str) -> Box:
        """Load YAML configuration file (delegated to cache)"""
        return self.model_cache._load_config(config_path)

    def _get_checkpoint_path(self, checkpoint_name: str) -> str:
        """Get checkpoint path, downloading if necessary"""
        if checkpoint_name not in self._checkpoint_cache:
            self.logger.info(f"Loading checkpoint: {checkpoint_name}")
            self._checkpoint_cache[checkpoint_name] = download(checkpoint_name)

        return self._checkpoint_cache[checkpoint_name]

    def _preprocess_input(
        self, input_path: str, cache_dir: str, faces_target_count: int = 50000
    ) -> str:
        """
        Preprocess a single input file to generate raw_data.npz and other required data

        Args:
            input_path: Path to input 3D model file
            cache_dir: Directory to store processed data
            faces_target_count: Target number of faces for mesh simplification

        Returns:
            Path to the directory containing processed data
        """
        self.logger.info(f"Preprocessing input file: {input_path}")

        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Check file extension (support common 3D formats but assume FBX for pipeline)
        valid_extensions = [".obj", ".fbx", ".FBX", ".dae", ".glb", ".gltf", ".vrm"]
        if not any(input_path.endswith(ext) for ext in valid_extensions):
            raise ValueError(
                f"Unsupported file format. Supported formats: {valid_extensions}"
            )

        # Generate output directory name (remove extension from filename)
        file_name = os.path.basename(input_path)
        file_name_without_ext = ".".join(file_name.split(".")[:-1])
        output_dir = os.path.join(cache_dir, file_name_without_ext)

        # Check if already processed
        raw_data_path = os.path.join(output_dir, "raw_data.npz")
        if os.path.exists(raw_data_path):
            self.logger.info(f"Found existing processed data: {raw_data_path}")
            return output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Use extract_builtin to process the file
        files = [(input_path, output_dir)]  # List of (input_file, output_dir) tuples

        try:
            extract_builtin(
                output_folder=cache_dir,
                target_count=faces_target_count,
                num_runs=1,
                id=0,
                time="inference",  # Use fixed time for inference
                files=files,
            )

            # Verify the output was created
            if not os.path.exists(raw_data_path):
                raise RuntimeError(
                    f"Preprocessing failed: {raw_data_path} was not created"
                )

            self.logger.info(f"Preprocessing completed: {raw_data_path}")
            return output_dir

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            # Clean up on failure
            if os.path.exists(output_dir):
                import shutil

                shutil.rmtree(output_dir, ignore_errors=True)
            raise

    def _run_inference_with_system(
        self,
        loaded_system: LoadedSystem,
        input_path: str,
        output_dir: str = "results",
        output_filename: Optional[str] = None,
        data_name: Optional[str] = None,
        cls: Optional[str] = None,
    ) -> str:
        """
        Shared inference logic using a pre-loaded system
        Assumes preprocessing has already been done for the input_path
        """
        # Use data_name from task config if not specified
        if data_name is None:
            data_name = loaded_system.task_config.components.data_name

        # Get processed data directory from input path
        file_name = os.path.basename(input_path)
        file_name_without_ext = ".".join(file_name.split(".")[:-1])
        processed_dir = os.path.join(self.config.cache_dir, file_name_without_ext)

        # Verify processed data exists
        processed_data_path = os.path.join(processed_dir, data_name)
        if not os.path.exists(processed_data_path):
            raise RuntimeError(
                f"Processed data not found: {processed_data_path}. Make sure preprocessing was called first."
            )

        files = [processed_dir]

        # Create datapath
        datapath = Datapath(files=files, cls=cls)

        # Prepare dataset configs (these are already parsed and cached)
        predict_dataset_config = loaded_system.data_config.get("predict_dataset_config")
        if predict_dataset_config:
            predict_dataset_config = DatasetConfig.parse(
                config=predict_dataset_config
            ).split_by_cls()

        predict_transform_config = loaded_system.transform_config.get(
            "predict_transform_config"
        )
        if predict_transform_config:
            predict_transform_config = TransformConfig.parse(
                config=predict_transform_config
            )

        # Create data module
        data_module = UniRigDatasetModule(
            process_fn=loaded_system.model._process_fn if loaded_system.model else None,
            predict_dataset_config=predict_dataset_config,
            predict_transform_config=predict_transform_config,
            tokenizer_config=TokenizerConfig.parse(
                config=loaded_system.tokenizer_config
            )
            if loaded_system.tokenizer_config
            else None,
            debug=False,
            data_name=data_name,
            datapath=datapath,
            cls=cls,
        )

        # Setup data module
        data_module.setup("predict")
        predict_dataloader = data_module.predict_dataloader()

        # Setup writer
        writer_config = loaded_system.writer_config.copy()
        writer_config["npz_dir"] = self.config.cache_dir
        writer_config["output_dir"] = output_dir
        writer_config["output_name"] = (
            output_filename  # Notice that `output_name` will overrides the output directory
        )
        writer_config["user_mode"] = (
            True  # Always use user mode for inference, which DOES NOT produce the .npz data
        )

        writer = get_writer(
            **writer_config, order_config=predict_transform_config.order_config
        )

        # Run inference with the cached system
        with self._inference_mode():
            for batch_idx, batch in enumerate(predict_dataloader[cls]):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)

                # Generate prediction
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    prediction = loaded_system.system.predict_step(batch, batch_idx)

                # Write results
                writer.write_on_batch_end(
                    trainer=None,
                    pl_module=loaded_system.system,
                    prediction=prediction,
                    batch_indices=[batch_idx],
                    batch=batch,
                    batch_idx=batch_idx,
                    dataloader_idx=0,
                )

        # Finalize writer
        # writer.on_predict_end(trainer=None, pl_module=loaded_system.system)

        # Return the output path
        if output_filename:
            return output_filename
        else:
            return output_dir

    @contextmanager
    def _inference_mode(self):
        """Context manager for inference mode"""
        was_training = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            yield
        finally:
            torch.set_grad_enabled(was_training)

    def generate_skeleton(
        self,
        input_path: str,
        output_dir: str = "results",
        output_filename: Optional[str] = None,
        skeleton_task_config: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate skeleton for input mesh (assumes FBX input)

        Args:
            input_path: Path to input FBX file
            output_dir: Directory to save output files
            output_filename: Specific output filename
            skeleton_task_config: Path to skeleton task configuration
            **kwargs: Additional configuration overrides

        Returns:
            Path to the generated skeleton file
        """
        self.logger.info("Starting skeleton generation")

        # Preprocess the input FBX to generate required data
        self.logger.info(
            f"Preprocessing input FBX for skeleton generation: {input_path}"
        )
        self._preprocess_input(input_path, self.config.cache_dir)

        # Use default config if not specified
        if skeleton_task_config is None:
            skeleton_task_config = self.default_skeleton_config

        # Get the loaded system (with checkpoint already loaded and cached)
        loaded_system = self.model_cache.get_loaded_system(
            skeleton_task_config, self.config.compile_model
        )

        # Run inference using the shared logic
        result = self._run_inference_with_system(
            loaded_system=loaded_system,
            input_path=input_path,
            output_dir=output_dir,
            output_filename=output_filename,
            cls=kwargs.get("cls", None),
        )

        self.logger.info("Skeleton generation completed")
        return result

    def generate_skin_weights(
        self,
        input_path: str,
        output_dir: str = "results",
        output_filename: Optional[str] = None,
        skin_task_config: Optional[str] = None,
        skeleton_data_name: str = "predict_skeleton.npz",
        **kwargs,
    ) -> str:
        """
        Generate skin weights for input mesh with existing skeleton (assumes FBX input)

        Args:
            input_path: Path to input FBX file (should be skeleton output from previous stage)
            output_dir: Directory to save output files
            output_filename: Specific output filename
            skin_task_config: Path to skin task configuration
            skeleton_data_name: Name of the skeleton data file from previous phase
            **kwargs: Additional configuration overrides

        Returns:
            Path to the generated skin weights file
        """
        self.logger.info("Starting skin weights generation")

        # Preprocess the input FBX (skeleton output) to generate required data for skin stage
        self.logger.info(
            f"Preprocessing skeleton FBX for skin generation: {input_path}"
        )
        self._preprocess_input(input_path, self.config.cache_dir)

        # Use default config if not specified
        if skin_task_config is None:
            skin_task_config = self.default_skin_config

        # Get the loaded system (with checkpoint already loaded and cached)
        loaded_system = self.model_cache.get_loaded_system(
            skin_task_config, self.config.compile_model
        )

        # Run inference using the shared logic
        result = self._run_inference_with_system(
            loaded_system=loaded_system,
            input_path=input_path,
            output_dir=output_dir,
            output_filename=output_filename,
            data_name="raw_data.npz",  # Use raw_data.npz since we preprocessed the skeleton FBX
            cls=kwargs.get("cls", None),
        )

        self.logger.info("Skin weights generation completed")
        return result

    def full_pipeline(
        self,
        input_path: str,
        output_dir: str = "results",
        output_filename: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Run the complete rigging pipeline: skeleton generation -> skin weights generation

        Args:
            input_path: Path to input FBX file
            output_dir: Directory to save output files
            output_filename: Specific output filename
            **kwargs: Additional configuration overrides

        Returns:
            Path to the final rigged model
        """
        self.logger.info("Starting full rigging pipeline")

        # Step 1: Generate skeleton (preprocesses input FBX internally)
        skeleton_result = self.generate_skeleton(
            input_path=input_path,
            output_dir=self.config.cache_dir,
            output_filename=output_filename,
            **kwargs,
        )

        # Step 2: Generate skin weights using skeleton output (preprocesses skeleton FBX internally)
        skin_result = self.generate_skin_weights(
            input_path=skeleton_result,
            output_dir=output_dir,
            output_filename=output_filename,
            **kwargs,
        )

        self.logger.info("Full rigging pipeline completed")
        return skin_result

    def clear_cache(self):
        """Clear all cached models and data"""
        self.model_cache.clear()
        self._checkpoint_cache.clear()

    def __del__(self):
        """Cleanup when engine is destroyed"""
        self.clear_cache()

    def preload_systems(
        self, skeleton_config: Optional[str] = None, skin_config: Optional[str] = None
    ):
        """
        Preload skeleton and skin systems to avoid loading during inference

        Args:
            skeleton_config: Path to skeleton task configuration
            skin_config: Path to skin task configuration
        """
        self.logger.info("Preloading inference systems...")

        # Preload skeleton system
        if skeleton_config is None:
            skeleton_config = self.default_skeleton_config
        self.logger.info(f"Loading skeleton system: {skeleton_config}")
        self.model_cache.get_loaded_system(skeleton_config, self.config.compile_model)

        # Preload skin system
        if skin_config is None:
            skin_config = self.default_skin_config
        self.logger.info(f"Loading skin system: {skin_config}")
        self.model_cache.get_loaded_system(skin_config, self.config.compile_model)

        self.logger.info("All systems preloaded successfully!")


# Convenience functions for quick usage
def create_inference_engine(
    device: str = "auto", compile_models: bool = True, cache_dir: str = "tmp"
) -> UniRigInferenceEngine:
    """Create a pre-configured inference engine"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = InferenceConfig(
        device=device, compile_model=compile_models, cache_dir=cache_dir
    )

    return UniRigInferenceEngine(config)
