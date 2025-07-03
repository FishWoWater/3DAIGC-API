"""
PartPacker utility for part-level 3D mesh generation from single-view images.

This module provides a clean interface to the PartPacker model adapted for our framework.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import kiui
import numpy as np
import torch
import trimesh

logger = logging.getLogger(__name__)


class PartPackerError(Exception):
    """Custom exception for PartPacker-related errors."""

    pass


class PartPackerRunner:
    """
    A utility class for PartPacker 3D mesh generation.

    This class encapsulates the PartPacker functionality for generating part-level 3D meshes
    from single-view images.
    """

    # GLB export transform matrix for trimesh
    TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)

    # Default configuration mappings
    DEFAULT_CONFIGS = {
        "big_parts_strict_pvae": "flow.configs.big_parts_strict_pvae",
        "default": "flow.configs.big_parts_strict_pvae",
    }

    def __init__(
        self,
        config_name: str = "default",
        flow_ckpt_path: str = "pretrained/PartPacker/flow.pt",
        device: Optional[str] = None,
        precision: str = "bfloat16",
        enable_background_removal: bool = True,
        partpacker_root: Optional[str] = None,
    ):
        """
        Initialize the PartPacker runner.

        Args:
            config_name: Name of the configuration to use
            flow_ckpt_path: Path to the flow model checkpoint
            device: Device to use for inference. If None, automatically selects GPU if available
            precision: Precision for inference ('bfloat16', 'float16', or 'float32')
            enable_background_removal: Whether to enable automatic background removal
            partpacker_root: Root directory of PartPacker code

        Raises:
            PartPackerError: If model loading fails or required files are missing
        """
        self.config_name = config_name
        self.flow_ckpt_path = flow_ckpt_path
        self.precision = precision
        self.enable_background_removal = enable_background_removal

        if partpacker_root is None:
            partpacker_root = "thirdparty/PartPacker"
        self.partpacker_root = Path(partpacker_root)

        # Add PartPacker to Python path
        if str(self.partpacker_root) not in sys.path:
            sys.path.insert(0, str(self.partpacker_root))

        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        logger.info(f"PartPacker using device: {self.device}")

        # Initialize background remover if enabled
        self.bg_remover = None
        if self.enable_background_removal:
            try:
                import rembg

                self.bg_remover = rembg.new_session()
                logger.info("Background removal enabled for PartPacker")
            except Exception as e:
                warnings.warn(f"Failed to initialize background remover: {e}")
                self.enable_background_removal = False

        # Model will be loaded when needed
        self.model = None
        self.config = None

        self._load_model()
        logger.info("PartPacker runner initialized")

    def _load_model(self) -> None:
        """Load the PartPacker model and weights."""
        if self.model is not None:
            return  # Already loaded

        try:
            # Validate checkpoint exists
            if not os.path.exists(self.flow_ckpt_path):
                raise PartPackerError(f"Checkpoint not found: {self.flow_ckpt_path}")

            logger.info(f"Loading PartPacker checkpoint from {self.flow_ckpt_path}")

            # Import PartPacker modules
            import importlib

            from flow.model import Model

            # Load checkpoint
            ckpt_dict = torch.load(
                self.flow_ckpt_path, map_location=self.device, weights_only=True
            )
            if "model" in ckpt_dict:
                ckpt_dict = ckpt_dict["model"]

            # Get configuration
            config_module = self.DEFAULT_CONFIGS.get(self.config_name, self.config_name)
            logger.info(f"Loading PartPacker configuration from {config_module}")

            try:
                model_config = importlib.import_module(config_module).make_config()
                # rewrite the vae config path
                model_config.vae_ckpt_path = "pretrained/PartPacker/vae.pt"
            except ImportError as e:
                raise PartPackerError(f"Failed to load config {config_module}: {e}")

            # Instantiate model
            self.model = Model(model_config).eval()

            # Set precision
            if self.precision == "bfloat16":
                self.model = self.model.bfloat16()
            elif self.precision == "float16":
                self.model = self.model.half()
            else:
                self.model = self.model.float()

            # Move to device
            self.model = self.model.to(self.device)

            # Load weights
            self.model.load_state_dict(ckpt_dict, strict=True)

            # Store config for reference
            self.config = model_config

            logger.info("PartPacker model loaded successfully")

        except Exception as e:
            raise PartPackerError(f"Failed to load PartPacker model: {e}")

    def preprocess_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        target_size: int = 518,
        border_ratio: float = 0.1,
    ) -> np.ndarray:
        """
        Preprocess an input image for the model.

        Args:
            image_path: Path to image file or numpy array
            target_size: Target size for the image (default: 518)
            border_ratio: Border ratio for recentering (default: 0.1)

        Returns:
            Preprocessed image as numpy array [H, W, 3]

        Raises:
            PartPackerError: If image loading or processing fails
        """
        try:
            from flow.utils import recenter_foreground

            # Load image
            if isinstance(image_path, (str, Path)):
                if not os.path.exists(image_path):
                    raise PartPackerError(f"Image not found: {image_path}")
                input_image = kiui.read_image(
                    str(image_path), mode="uint8", order="RGBA"
                )
            elif isinstance(image_path, np.ndarray):
                input_image = image_path
                if input_image.shape[-1] == 3:
                    # Add alpha channel
                    alpha = (
                        np.ones((*input_image.shape[:2], 1), dtype=input_image.dtype)
                        * 255
                    )
                    input_image = np.concatenate([input_image, alpha], axis=-1)
            else:
                raise PartPackerError("Invalid image input type")

            # Background removal if no alpha channel or if enabled
            if input_image.shape[-1] == 3 or (
                self.enable_background_removal and self.bg_remover is not None
            ):
                if input_image.shape[-1] == 3:
                    # Add dummy alpha for processing
                    alpha = (
                        np.ones((*input_image.shape[:2], 1), dtype=input_image.dtype)
                        * 255
                    )
                    input_image = np.concatenate([input_image, alpha], axis=-1)

                if self.bg_remover is not None:
                    import rembg

                    input_image = rembg.remove(input_image, session=self.bg_remover)

            # Create mask and recenter
            mask = input_image[..., -1] > 0
            if not mask.any():
                warnings.warn("No foreground detected in image")
                # Create a dummy mask for the entire image
                mask = np.ones(input_image.shape[:2], dtype=bool)

            image = recenter_foreground(input_image, mask, border_ratio=border_ratio)

            # Resize to target size
            image = cv2.resize(
                image, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )

            # Convert to float and apply background
            image = image.astype(np.float32) / 255.0
            image = image[..., :3] * image[..., 3:4] + (
                1 - image[..., 3:4]
            )  # white background

            return image

        except Exception as e:
            raise PartPackerError(f"Failed to preprocess image: {e}")

    @torch.inference_mode()
    def generate_from_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        num_steps: int = 30,
        cfg_scale: float = 7.0,
        grid_resolution: int = 384,
        num_faces: int = 50000,
        seed: Optional[int] = None,
        return_parts: bool = True,
        return_volumes: bool = True,
    ) -> Dict:
        """
        Generate a 3D mesh from a single-view image.

        Args:
            image_path: Path to input image or numpy array
            num_steps: Number of diffusion steps (default: 30)
            cfg_scale: Classifier-free guidance scale (default: 7.0)
            grid_resolution: Grid resolution for mesh extraction (default: 384)
            num_faces: Target number of faces for decimation (default: 50000)
            seed: Random seed for reproducibility
            return_parts: Whether to return individual parts (default: True)
            return_volumes: Whether to return dual volumes (default: True)

        Returns:
            Dictionary containing generated meshes and metadata

        Raises:
            PartPackerError: If generation fails
        """
        try:
            # Ensure model is loaded
            self._load_model()

            from flow.utils import get_random_color
            from vae.utils import postprocess_mesh

            # Set seed if provided
            if seed is not None:
                kiui.seed_everything(seed)

            # Preprocess image
            image = self.preprocess_image(image_path)

            # Convert to tensor
            image_tensor = (
                torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0)
            )
            image_tensor = image_tensor.to(self.device)

            # Prepare model input
            data = {"cond_images": image_tensor}

            # Run model
            logger.info("Running PartPacker model inference...")
            results = self.model(data, num_steps=num_steps, cfg_scale=cfg_scale)
            latent = results["latent"]

            # Generate meshes
            output = {
                "preprocessed_image": image,
                "parts": [],
                "combined_mesh": None,
                "dual_volumes": [],
                "metadata": {
                    "num_steps": num_steps,
                    "cfg_scale": cfg_scale,
                    "grid_resolution": grid_resolution,
                    "num_faces": num_faces,
                    "seed": seed,
                    "use_parts": self.config.use_parts,
                },
            }

            if self.config.use_parts:
                # Split latent into two parts
                data_part0 = {"latent": latent[:, : self.config.latent_size, :]}
                data_part1 = {"latent": latent[:, self.config.latent_size :, :]}

                # Generate meshes for each part
                results_part0 = self.model.vae(data_part0, resolution=grid_resolution)
                results_part1 = self.model.vae(data_part1, resolution=grid_resolution)

                # Process part 0
                vertices, faces = results_part0["meshes"][0]
                mesh_part0 = trimesh.Trimesh(vertices, faces)
                mesh_part0.vertices = mesh_part0.vertices @ self.TRIMESH_GLB_EXPORT.T
                mesh_part0 = postprocess_mesh(mesh_part0, num_faces)

                # Process part 1
                vertices, faces = results_part1["meshes"][0]
                mesh_part1 = trimesh.Trimesh(vertices, faces)
                mesh_part1.vertices = mesh_part1.vertices @ self.TRIMESH_GLB_EXPORT.T
                mesh_part1 = postprocess_mesh(mesh_part1, num_faces)

                # Split into connected components
                parts = mesh_part0.split(only_watertight=False)
                if isinstance(parts, np.ndarray):
                    parts = parts.tolist()
                splitted_mesh_part1 = mesh_part1.split(only_watertight=False)
                if isinstance(splitted_mesh_part1, np.ndarray):
                    splitted_mesh_part1 = splitted_mesh_part1.tolist()
                parts.extend(splitted_mesh_part1)

                # Assign colors to parts
                if return_parts:
                    for j, part in enumerate(parts):
                        part.visual.vertex_colors = get_random_color(j, use_float=True)
                    output["parts"] = parts

                # Create combined mesh
                output["combined_mesh"] = trimesh.Scene(parts)

                # Store dual volumes
                if return_volumes:
                    output["dual_volumes"] = [mesh_part0, mesh_part1]

            else:
                # Single mesh generation
                data = {"latent": latent}
                results = self.model.vae(data, resolution=grid_resolution)

                vertices, faces = results["meshes"][0]
                mesh = trimesh.Trimesh(vertices, faces)
                mesh.vertices = mesh.vertices @ self.TRIMESH_GLB_EXPORT.T
                mesh = postprocess_mesh(mesh, num_faces)

                output["combined_mesh"] = mesh

            logger.info("PartPacker mesh generation completed successfully")
            return output

        except Exception as e:
            raise PartPackerError(f"Failed to generate mesh with PartPacker: {e}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "config_name": self.config_name,
            "device": self.device,
            "precision": self.precision,
            "use_parts": self.config.use_parts if self.config else None,
            "latent_size": self.config.latent_size if self.config else None,
            "latent_dim": self.config.latent_dim if self.config else None,
            "background_removal_enabled": self.enable_background_removal,
        }

    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.bg_remover is not None:
            del self.bg_remover
            self.bg_remover = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("PartPacker runner cleaned up")
