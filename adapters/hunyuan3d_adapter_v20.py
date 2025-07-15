"""
Hunyuan3D 2.0 model adapters for image-to-mesh generation and mesh painting.

This adapter integrates Hunyuan3D 2.0 models into our mesh generation framework,
supporting both shape generation and texture painting workflows.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from core.models.base import ModelStatus
from core.models.mesh_models import ImageToMeshModel
from core.utils.thumbnail_utils import generate_mesh_thumbnail
from utils.file_utils import OutputPathGenerator
from utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class Hunyuan3DV20ImageToMeshAdapterCommon(ImageToMeshModel):
    """
    Common base adapter for Hunyuan3D 2.0 image-to-mesh models.

    Provides shared functionality for shape generation and texture painting.
    """

    FEATURE_TYPE = "image_to_mesh"
    MODEL_ID = "hunyuan3d-2.0"

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        vram_requirement: int = 5120,  # 5GB VRAM base
        hunyuan3d_root: Optional[str] = None,
        feature_type: Optional[str] = None,
        supported_output_formats: Optional[List[str]] = None,
    ):
        if model_id is None:
            model_id = self.MODEL_ID
        if model_path is None:
            model_path = "tencent/Hunyuan3D-2"

        if hunyuan3d_root is None:
            hunyuan3d_root = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "Hunyuan3D-2")
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

        self.model_path = model_path
        self.model_path_mini = model_path.rstrip("/") + "mini"
        self.hunyuan3d_root = Path(hunyuan3d_root)
        self.pipeline_shapegen = None
        self.paint_pipeline = None
        self.bg_remover = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

        # Configuration for what to load (overridden by subclasses)
        self.load_shapegen = True
        self.load_painting = True

        # Add Hunyuan3D to Python path
        if str(self.hunyuan3d_root) not in sys.path:
            sys.path.insert(0, str(self.hunyuan3d_root))

        # We also need to setup the envar for pretrained models to load correctly
        os.environ["HY3DGEN_MODELS"] = str(
            (Path(os.getcwd()) / "pretrained").absolute()
        )

    def _load_model(self):
        """Load Hunyuan3D 2.0 pipelines based on configuration."""
        try:
            logger.info(f"Loading Hunyuan3D 2.0 models from {self.model_path}")

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
                from hy3dgen.rembg import BackgroundRemover
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

                logger.info("Loading shape generation pipeline...")
                self.pipeline_shapegen = (
                    Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                        self.model_path_mini,
                        subfolder="hunyuan3d-dit-v2-mini-turbo",
                        use_safetensors=False,
                        device="cuda",
                    )
                )
                self.pipeline_shapegen.enable_flashvdm(topk_mode="merge")
                loaded_models["shapegen"] = self.pipeline_shapegen

                # Load background remover
                logger.info("Loading background remover...")
                self.bg_remover = BackgroundRemover()
                loaded_models["bg_remover"] = self.bg_remover

            # Load paint pipeline if needed
            if self.load_painting:
                from hy3dgen.rembg import BackgroundRemover
                from hy3dgen.texgen import Hunyuan3DPaintPipeline

                logger.info("Loading paint pipeline...")
                self.paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                    self.model_path,
                    subfolder="hunyuan3d-paint-v2-0-turbo",
                )
                loaded_models["paint"] = self.paint_pipeline
                if "bg_remover" not in loaded_models:
                    self.bg_remover = BackgroundRemover()
                    loaded_models["bg_remover"] = self.bg_remover

            logger.info("Hunyuan3D 2.0 models loaded successfully")
            return loaded_models

        except Exception as e:
            logger.error(f"Failed to load Hunyuan3D 2.0 models: {str(e)}")
            raise Exception(f"Failed to load Hunyuan3D 2.0 models: {str(e)}")

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

            logger.info("Hunyuan3D 2.0 models unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading Hunyuan3D models: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-mesh generation using Hunyuan3D 2.0.

        Should be overridden by subclasses for specific workflows.
        """
        raise NotImplementedError("Subclasses must implement _process_request")

    def _generate_output_path(self, base_name: str, output_format: str) -> Path:
        """Generate output file path."""
        return self.path_generator.generate_mesh_path(
            self.model_id, base_name, output_format
        )

    def _generate_thumbnail_path(self, mesh_path: Path) -> Path:
        """Generate thumbnail file path based on mesh path."""
        # Create thumbnails directory
        thumbnail_dir = Path(os.getcwd()) / "outputs" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        # Generate thumbnail filename
        thumbnail_name = mesh_path.stem + "_thumb.png"
        return thumbnail_dir / thumbnail_name

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for Hunyuan3D 2.0."""
        return {"input": ["png", "jpg", "jpeg"], "output": ["glb", "obj"]}


class Hunyuan3DV20ImageToRawMeshAdapter(Hunyuan3DV20ImageToMeshAdapterCommon):
    """
    Adapter for Hunyuan3D 2.0 image-to-raw-mesh generation.

    Only performs shape generation without texture painting.
    """

    FEATURE_TYPE = "image_to_raw_mesh"
    MODEL_ID = "hunyuan3d_2_0_image_to_raw_mesh"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_shapegen = True
        self.load_painting = False  # Don't load painting pipeline
        self.vram_requirement = 5120  # 5GB VRAM for shape only
        self.supported_output_formats = ["glb", "obj", "ply"]

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-raw-mesh generation using Hunyuan3D 2.0.
        """
        try:
            # Validate pipelines are loaded
            if self.bg_remover is None:
                raise ValueError("Background remover is not loaded")
            if self.pipeline_shapegen is None:
                raise ValueError("Shape generation pipeline is not loaded")

            # Validate inputs
            if "image_path" not in inputs:
                raise ValueError("image_path is required for image-to-mesh generation")

            image_path = Path(inputs["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Input image file not found: {image_path}")

            # Extract parameters
            output_format = inputs.get("output_format", "glb")
            num_inference_steps = inputs.get("num_inference_steps", 50)
            octree_resolution = inputs.get("octree_resolution", 380)
            num_chunks = inputs.get("num_chunks", 20000)
            seed = inputs.get("seed", 12345)

            if output_format not in self.supported_output_formats:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(
                f"Generating raw mesh with Hunyuan3D 2.0 from image: {image_path}"
            )

            # Load and preprocess image
            image = Image.open(image_path).convert("RGBA")
            if image.mode == "RGB":
                image = self.bg_remover(image)

            # Shape generation only
            logger.info("Generating 3D shape...")

            generator = torch.manual_seed(seed) if seed is not None else None

            mesh_result = self.pipeline_shapegen(
                image=image,
                num_inference_steps=num_inference_steps,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                generator=generator,
                output_type="trimesh",
            )[0]

            # Generate output path
            base_name = f"{self.model_id}_{image_path.stem}"
            output_path = self._generate_output_path(base_name, output_format)

            # Save raw mesh
            self.mesh_processor.save_mesh(mesh_result, output_path)

            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail_path(output_path)
            thumbnail_generated = generate_mesh_thumbnail(
                str(output_path), str(thumbnail_path)
            )

            # Load final mesh for statistics
            final_mesh = self.mesh_processor.load_mesh(output_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(final_mesh)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_generated else None,
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_image": str(image_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "has_texture": False,
                    "num_inference_steps": num_inference_steps,
                    "octree_resolution": octree_resolution,
                    "seed": seed,
                    "thumbnail_generated": thumbnail_generated,
                },
            }

            logger.info(f"Hunyuan3D 2.0 raw mesh generation completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"Hunyuan3D 2.0 raw mesh generation failed: {str(e)}")
            raise Exception(f"Hunyuan3D 2.0 raw mesh generation failed: {str(e)}")


class Hunyuan3DV20ImageToTexturedMeshAdapter(Hunyuan3DV20ImageToMeshAdapterCommon):
    """
    Adapter for Hunyuan3D 2.0 image-to-textured-mesh generation.

    Performs both shape generation and texture painting.
    """

    FEATURE_TYPE = "image_to_textured_mesh"
    MODEL_ID = "hunyuan3d_2_0_image_to_textured_mesh"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_shapegen = True
        self.load_painting = True  # Load both pipelines
        self.vram_requirement = 14336  # 14GB VRAM for full pipeline
        self.supported_output_formats = ["glb", "obj"]

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-textured-mesh generation using Hunyuan3D 2.0.
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
            num_inference_steps = inputs.get("num_inference_steps", 50)
            octree_resolution = inputs.get("octree_resolution", 380)
            num_chunks = inputs.get("num_chunks", 20000)
            seed = inputs.get("seed", 12345)

            if output_format not in self.supported_output_formats:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(
                f"Generating textured mesh with Hunyuan3D 2.0 from image: {image_path}"
            )

            # Load and preprocess image
            image = Image.open(image_path).convert("RGBA")
            if image.mode == "RGB":
                image = self.bg_remover(image)

            # Step 1: Shape generation
            logger.info("Generating 3D shape...")

            generator = torch.manual_seed(seed) if seed is not None else None
            mesh_result = self.pipeline_shapegen(
                image=image,
                num_inference_steps=num_inference_steps,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                generator=generator,
                output_type="trimesh",
            )[0]

            # Step 2: Texture painting
            logger.info("Generating texture...")

            final_mesh = self.paint_pipeline(mesh_result, image=image)

            # Generate output path
            base_name = f"{self.model_id}_{image_path.stem}"
            output_path = self._generate_output_path(base_name, output_format)

            # Save final mesh
            final_mesh.export(str(output_path))

            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail_path(output_path)
            thumbnail_generated = generate_mesh_thumbnail(
                str(output_path), str(thumbnail_path)
            )

            # Load final mesh for statistics
            loaded_mesh = self.mesh_processor.load_mesh(output_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(loaded_mesh)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_generated else None,
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_image": str(image_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "has_texture": True,
                    "num_inference_steps": num_inference_steps,
                    "octree_resolution": octree_resolution,
                    "seed": seed,
                    "thumbnail_generated": thumbnail_generated,
                },
            }

            logger.info(
                f"Hunyuan3D 2.0 textured mesh generation completed: {output_path}"
            )
            return response

        except Exception as e:
            logger.error(f"Hunyuan3D 2.0 textured mesh generation failed: {str(e)}")
            raise Exception(f"Hunyuan3D 2.0 textured mesh generation failed: {str(e)}")


class Hunyuan3DV20ImageMeshPaintingAdapter(Hunyuan3DV20ImageToMeshAdapterCommon):
    """
    Adapter for Hunyuan3D 2.0 mesh texture painting.

    Takes an existing mesh and reference image to generate textured mesh.
    Only loads the painting pipeline.
    """

    FEATURE_TYPE = "image_mesh_painting"
    MODEL_ID = "hunyuan3d_2_0_image_mesh_painting"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_shapegen = False  # Don't load shape generation pipeline
        self.load_painting = True  # Only load painting pipeline
        self.vram_requirement = 11264  # 11GB VRAM for painting only
        self.supported_output_formats = ["glb", "obj"]

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process mesh texture painting using Hunyuan3D 2.0.

        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - image_path: Path to reference image (required)
                - output_format: Output format (default: "glb")

        Returns:
            Dictionary with painting results
        """
        try:
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

            if output_format not in self.supported_output_formats:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(
                f"Painting mesh with Hunyuan3D 2.0: {mesh_path} using image: {image_path}"
            )

            # Load input mesh and image
            import trimesh

            mesh = trimesh.load(str(mesh_path))
            image = Image.open(image_path).convert("RGBA")
            if image.mode == "RGB" and self.bg_remover is not None:
                image = self.bg_remover(image)

            # Generate output path
            base_name = f"{self.model_id}_{mesh_path.stem}_{image_path.stem}"
            output_path = self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            )

            # Run texture painting
            logger.info("Generating texture...")

            final_mesh = self.paint_pipeline(mesh, image=image)

            # Save the textured mesh
            final_mesh.export(str(output_path))

            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail_path(output_path)
            thumbnail_generated = generate_mesh_thumbnail(
                str(output_path), str(thumbnail_path)
            )

            # Load final mesh for statistics
            loaded_mesh = self.mesh_processor.load_mesh(output_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(loaded_mesh)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "thumbnail_path": str(thumbnail_path) if thumbnail_generated else None,
                "success": True,
                "painting_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "input_image": str(image_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "thumbnail_generated": thumbnail_generated,
                },
            }

            logger.info(f"Hunyuan3D 2.0 mesh painting completed: {output_path}")
            return response

        except Exception as e:
            logger.error(f"Hunyuan3D 2.0 mesh painting failed: {str(e)}")
            raise Exception(f"Hunyuan3D 2.0 mesh painting failed: {str(e)}")

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for Hunyuan3D 2.0 painting."""
        return {"input": ["glb", "obj", "ply"], "output": ["glb", "obj"]}
