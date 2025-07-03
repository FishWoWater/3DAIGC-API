"""
Hunyuan3D 2.1 model adapters for image-to-mesh generation and mesh painting.

This adapter integrates Hunyuan3D 2.1 models into our mesh generation framework,
supporting both shape generation and texture painting workflows with proper variants.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from core.models.base import ModelStatus
from core.models.mesh_models import ImageToMeshModel
from utils.file_utils import OutputPathGenerator
from utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class Hunyuan3DV21ImageToMeshAdapterCommon(ImageToMeshModel):
    """
    Common base adapter for Hunyuan3D 2.1 image-to-mesh models.

    Provides shared functionality for shape generation and texture painting.
    """

    FEATURE_TYPE = "image_to_mesh"
    MODEL_ID = "hunyuan3d-2.1"

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        vram_requirement: int = 8192,  # 8GB VRAM base
        hunyuan3d_root: Optional[str] = None,
        feature_type: Optional[str] = None,
        supported_output_formats: Optional[List[str]] = None,
    ):
        if model_id is None:
            model_id = self.MODEL_ID
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.getcwd(), "pretrained", "tencent", "Hunyuan3D-2.1")
            )
        if hunyuan3d_root is None:
            hunyuan3d_root = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "Hunyuan3D-2.1")
            )
        if feature_type is None:
            feature_type = self.FEATURE_TYPE
        if supported_output_formats is None:
            supported_output_formats = ["glb", "obj"]

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=supported_output_formats,
            feature_type=feature_type,
        )

        self.hunyuan3d_root = Path(hunyuan3d_root)
        self.pipeline_shapegen = None
        self.paint_pipeline = None
        self.bg_remover = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

        # Configuration for what to load (overridden by subclasses)
        self.load_shapegen = True
        self.load_painting = True

        self.model_path = Path(model_path)
        # Add Hunyuan3D to Python path
        if str(self.hunyuan3d_root) not in sys.path:
            sys.path.insert(0, str(self.hunyuan3d_root))
        if str(self.hunyuan3d_root / "hy3dshape") not in sys.path:
            sys.path.insert(0, str(self.hunyuan3d_root / "hy3dshape"))

    def _load_model(self):
        """Load Hunyuan3D 2.1 pipelines based on configuration."""
        try:
            logger.info(f"Loading Hunyuan3D 2.1 models from {self.model_path}")

            # Apply torchvision fix if available
            try:
                from utils.torchvision_fix import apply_fix

                apply_fix()
            except ImportError:
                logger.warning(
                    "torchvision_fix module not found, proceeding without compatibility fix"
                )
            except Exception as e:
                logger.warning(f"Failed to apply torchvision fix: {e}")

            loaded_models = {}

            # Load shape generation pipeline if needed
            if self.load_shapegen:
                from hy3dshape.pipelines import (
                    Hunyuan3DDiTFlowMatchingPipeline,
                )
                from hy3dshape.rembg import BackgroundRemover

                logger.info("Loading shape generation pipeline...")
                self.pipeline_shapegen = (
                    Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_path)
                )
                loaded_models["shapegen"] = self.pipeline_shapegen

                # Load background remover
                logger.info("Loading background remover...")
                self.bg_remover = BackgroundRemover()
                loaded_models["bg_remover"] = self.bg_remover

            # Load paint pipeline if needed
            if self.load_painting:
                from hy3dpaint.textureGenPipeline import (
                    Hunyuan3DPaintConfig,
                    Hunyuan3DPaintPipeline,
                )
                from hy3dshape.rembg import BackgroundRemover

                logger.info("Setting up paint pipeline...")
                max_num_view = 6
                resolution = 512
                conf = Hunyuan3DPaintConfig(max_num_view, resolution)

                conf.multiview_cfg_path = str(
                    self.hunyuan3d_root / "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
                )
                conf.custom_pipeline = str(
                    self.hunyuan3d_root / "hy3dpaint/hunyuanpaintpbr"
                )
                conf.multiview_pretrained_path = str(self.model_path)
                conf.dino_ckpt_path = str(
                    self.model_path / ".." / ".." / "dinov2-giant"
                )
                conf.realesrgan_ckpt_path = str(
                    self.model_path / ".." / ".." / "misc" / "RealESRGAN_x4plus.pth"
                )

                if "bg_remover" not in loaded_models:
                    self.bg_remover = BackgroundRemover()
                    loaded_models["bg_remover"] = self.bg_remover

                self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
                loaded_models["paint"] = self.paint_pipeline

            logger.info("Hunyuan3D 2.1 models loaded successfully")
            return loaded_models

        except Exception as e:
            logger.error(f"Failed to load Hunyuan3D 2.1 models: {str(e)}")
            raise Exception(f"Failed to load Hunyuan3D 2.1 models: {str(e)}")

    def _unload_model(self):
        """Unload Hunyuan3D models."""
        try:
            if self.pipeline_shapegen is not None:
                del self.pipeline_shapegen
                self.pipeline_shapegen = None

            if self.paint_pipeline is not None:
                del self.paint_pipeline
                self.paint_pipeline = None

            if self.bg_remover is not None:
                del self.bg_remover
                self.bg_remover = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Hunyuan3D 2.1 models unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading Hunyuan3D models: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-mesh generation using Hunyuan3D 2.1.

        Should be overridden by subclasses for specific workflows.
        """
        raise NotImplementedError("Subclasses must implement _process_request")

    def _generate_output_path(self, base_name: str, output_format: str) -> Path:
        """Generate output file path."""
        return self.path_generator.generate_mesh_path(
            self.model_id, base_name, output_format
        )

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for Hunyuan3D 2.1."""
        return {"input": ["png", "jpg", "jpeg"], "output": ["glb", "obj"]}


class Hunyuan3DV21ImageToRawMeshAdapter(Hunyuan3DV21ImageToMeshAdapterCommon):
    """
    Adapter for Hunyuan3D 2.1 image-to-raw-mesh generation.

    Only performs shape generation without texture painting.
    """

    FEATURE_TYPE = "image_to_raw_mesh"
    MODEL_ID = "hunyuan3d_image_to_raw_mesh"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_shapegen = True
        self.load_painting = False  # Don't load painting pipeline
        self.vram_requirement = 8192  # 8GB VRAM for shape only
        self.supported_output_formats = ["glb", "obj", "ply"]

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-raw-mesh generation using Hunyuan3D 2.1.
        """
        try:
            # Validate inputs
            if "image_path" not in inputs:
                raise ValueError("image_path is required for image-to-mesh generation")

            image_path = Path(inputs["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Input image file not found: {image_path}")

            # Extract parameters
            output_format = inputs.get("output_format", "glb")

            if output_format not in self.supported_output_formats:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(
                f"Generating raw mesh with Hunyuan3D 2.1 from image: {image_path}"
            )

            # Load and preprocess image
            image = Image.open(image_path).convert("RGBA")
            if image.mode == "RGB":
                image = self.bg_remover(image)

            # Shape generation only
            logger.info("Generating 3D shape...")
            mesh_result = self.pipeline_shapegen(image=image)[0]

            # Generate output path
            base_name = f"{self.model_id}_{image_path.stem}"
            output_path = self._generate_output_path(base_name, output_format)

            # Save raw mesh
            mesh_result.export(str(output_path))

            # Load final mesh for statistics
            final_mesh = self.mesh_processor.load_mesh(output_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(final_mesh)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_image": str(image_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "has_texture": False,
                },
            }

            logger.info(f"Hunyuan3D 2.1 raw mesh generation completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"Hunyuan3D 2.1 raw mesh generation failed: {str(e)}")
            raise Exception(f"Hunyuan3D 2.1 raw mesh generation failed: {str(e)}")


class Hunyuan3DV21ImageToTexturedMeshAdapter(Hunyuan3DV21ImageToMeshAdapterCommon):
    """
    Adapter for Hunyuan3D 2.1 image-to-textured-mesh generation.

    Performs both shape generation and texture painting.
    """

    FEATURE_TYPE = "image_to_textured_mesh"
    MODEL_ID = "hunyuan3d_image_to_textured_mesh"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_shapegen = True
        self.load_painting = True  # Load both pipelines
        self.vram_requirement = 19456  # 19GB VRAM for full pipeline
        self.supported_output_formats = ["glb", "obj"]

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-textured-mesh generation using Hunyuan3D 2.1.
        """
        try:
            if self.bg_remover is None:
                raise ValueError("Background remover is not loaded")
            if self.pipeline_shapegen is None:
                raise ValueError("Shape generation pipeline is not loaded")
            if self.paint_pipeline is None:
                raise ValueError("Texture painting pipeline is not loaded")

            # Validate inputs
            if "image_path" not in inputs:
                raise ValueError("image_path is required for image-to-mesh generation")

            image_path = Path(inputs["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Input image file not found: {image_path}")

            # Extract parameters
            output_format = inputs.get("output_format", "glb")
            max_num_view = inputs.get("max_num_view", 6)
            resolution = inputs.get("resolution", 512)

            if output_format not in self.supported_output_formats:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(
                f"Generating textured mesh with Hunyuan3D 2.1 from image: {image_path}"
            )

            # Load and preprocess image
            image = Image.open(image_path).convert("RGBA")
            if image.mode == "RGB":
                image = self.bg_remover(image)

            # Step 1: Shape generation
            logger.info("Generating 3D shape...")
            mesh_result = self.pipeline_shapegen(image=image)[0]

            # Step 2: Texture painting
            logger.info("Generating texture...")

            # Save intermediate shape mesh
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                temp_mesh_path = tmp.name
                mesh_result.export(temp_mesh_path)

            try:
                # Update paint pipeline config if needed
                if hasattr(self.paint_pipeline.config, "max_num_view"):
                    self.paint_pipeline.config.max_num_view = max_num_view
                if hasattr(self.paint_pipeline.config, "resolution"):
                    self.paint_pipeline.config.resolution = resolution

                # Generate final output path
                base_name = f"{self.model_id}_{image_path.stem}"
                output_path = self._generate_output_path(base_name, output_format)

                # Run texture painting
                final_mesh_path = self.paint_pipeline(
                    temp_mesh_path, str(image_path), str(output_path)
                )

                # Ensure the output is at our desired path
                if final_mesh_path != str(output_path):
                    import shutil

                    shutil.move(final_mesh_path, output_path)

            finally:
                # Clean up temporary file
                Path(temp_mesh_path).unlink(missing_ok=True)

            # Load final mesh for statistics
            final_mesh = self.mesh_processor.load_mesh(output_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(final_mesh)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_image": str(image_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "has_texture": True,
                    "max_num_view": max_num_view,
                    "resolution": resolution,
                },
            }

            logger.info(
                f"Hunyuan3D 2.1 textured mesh generation completed: {output_path}"
            )
            return response

        except Exception as e:
            logger.error(f"Hunyuan3D 2.1 textured mesh generation failed: {str(e)}")
            raise Exception(f"Hunyuan3D 2.1 textured mesh generation failed: {str(e)}")


class Hunyuan3DV21ImageMeshPaintingAdapter(Hunyuan3DV21ImageToMeshAdapterCommon):
    """
    Adapter for Hunyuan3D 2.1 mesh texture painting.

    Takes an existing mesh and reference image to generate textured mesh.
    Only loads the painting pipeline.
    """

    FEATURE_TYPE = "image_mesh_painting"
    MODEL_ID = "hunyuan3d_image_mesh_painting"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_shapegen = False  # Don't load shape generation pipeline
        self.load_painting = True  # Only load painting pipeline
        self.vram_requirement = 12288  # 12GB VRAM for painting only
        self.supported_output_formats = ["glb", "obj"]

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process mesh texture painting using Hunyuan3D 2.1.

        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - image_path: Path to reference image (required)
                - output_format: Output format (default: "glb")
                - max_num_view: Number of views for texture generation (default: 6)
                - resolution: Texture resolution (default: 512)

        Returns:
            Dictionary with painting results
        """
        try:
            if self.bg_remover is None:
                raise ValueError("Background remover is not loaded")
            if self.paint_pipeline is None:
                raise ValueError("Texture painting pipeline is not loaded")

            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for mesh painting")
            if "image_path" not in inputs:
                raise ValueError("image_path is required for mesh painting")

            mesh_path = Path(inputs["mesh_path"])
            image_path = Path(inputs["image_path"])

            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")
            if not image_path.exists():
                raise FileNotFoundError(f"Input image file not found: {image_path}")

            # Extract parameters
            output_format = inputs.get("output_format", "glb")
            max_num_view = inputs.get("max_num_view", 6)
            resolution = inputs.get("resolution", 512)

            if output_format not in self.supported_output_formats:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(
                f"Painting mesh with Hunyuan3D 2.1: {mesh_path} using image: {image_path}"
            )

            # Update paint pipeline config
            if hasattr(self.paint_pipeline.config, "max_num_view"):
                self.paint_pipeline.config.max_num_view = max_num_view
            if hasattr(self.paint_pipeline.config, "resolution"):
                self.paint_pipeline.config.resolution = resolution

            # Generate output path
            base_name = f"{self.model_id}_{mesh_path.stem}_{image_path.stem}"
            output_path = self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            )

            # Run texture painting
            logger.info("Generating texture...")
            final_mesh_path = self.paint_pipeline(
                str(mesh_path), str(image_path), str(output_path)
            )

            # Ensure the output is at our desired path
            if final_mesh_path != str(output_path):
                import shutil

                shutil.move(final_mesh_path, output_path)

            # Load final mesh for statistics
            final_mesh = self.mesh_processor.load_mesh(output_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(final_mesh)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "success": True,
                "painting_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "input_image": str(image_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "max_num_view": max_num_view,
                    "resolution": resolution,
                },
            }

            logger.info(f"Hunyuan3D 2.1 mesh painting completed: {output_path}")
            return response

        except Exception as e:
            logger.error(f"Hunyuan3D 2.1 mesh painting failed: {str(e)}")
            raise Exception(f"Hunyuan3D 2.1 mesh painting failed: {str(e)}")

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for Hunyuan3D 2.1 painting."""
        return {"input": ["glb", "obj", "ply"], "output": ["glb", "obj"]}
