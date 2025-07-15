"""
TRELLIS model adapter for text-to-mesh generation.

This adapter integrates the TRELLIS model into our mesh generation framework.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import trimesh
from PIL import Image

from core.models.base import ModelStatus
from core.models.mesh_models import ImageToMeshModel, TextToMeshModel
from core.utils.thumbnail_utils import generate_mesh_thumbnail
from utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)

"""
NOTE: Mesh Painting in TRELLIS expects the mesh in Z-Up conventions
"""


class TrellisTextToMeshAdapterCommon(TextToMeshModel):
    """
    Adapter for TRELLIS text-to-mesh model.

    Integrates the TRELLIS model from the thirdparty/TRELLIS directory
    into our standardized mesh generation framework.
    """

    FEATURE_TYPE = "text_to_textured_mesh"  # Feature type for this adapter
    MODEL_ID = "trellis_text_to_textured_mesh"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vram_requirement: int = 11776,  # 12GB VRAM
        trellis_root: Optional[str] = None,
    ):
        # Set default paths
        if model_path is None:
            model_path = os.path.abspath(
                os.path.join(os.getcwd(), "pretrained", "TRELLIS")
            )

        if trellis_root is None:
            trellis_root = os.path.abspath(
                os.path.join(os.getcwd(), "thirdparty", "TRELLIS")
            )

        super().__init__(
            model_id=self.MODEL_ID,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=["glb", "obj"],
            feature_type=self.FEATURE_TYPE,
        )

        self.trellis_root = Path(trellis_root)
        self.model_path = Path(model_path)
        self.skip_models = [
            "slat_decoder_rf"
        ]  # Skip some models conditionally to save VRAM
        self.pipeline = None
        self.mesh_processor = MeshProcessor()

        # Add TRELLIS to Python path if not already there
        if str(self.trellis_root) not in sys.path:
            sys.path.insert(0, str(self.trellis_root))

    def _load_model(self):
        """Load the TRELLIS model pipeline."""
        try:
            logger.info(f"Loading TRELLIS model from {self.trellis_root}")

            # Import TRELLIS modules
            from trellis.pipelines import TrellisTextTo3DPipeline
            from trellis.utils import postprocessing_utils

            # Initialize the pipeline
            self.pipeline: TrellisTextTo3DPipeline = (
                TrellisTextTo3DPipeline.from_pretrained(
                    "microsoft/TRELLIS-text-xlarge",
                    cache_dir=str(self.model_path / "TRELLIS-text-xlarge"),
                )
            )
            self.pipeline.cuda()
            self.postprocessing_utils = postprocessing_utils

            logger.info("TRELLIS model loaded successfully")
            return self.pipeline

        except Exception as e:
            logger.error(f"Failed to load TRELLIS model: {str(e)}")
            raise Exception(f"Failed to load TRELLIS model: {str(e)}")

    def _unload_model(self):
        """Unload the TRELLIS model."""
        try:
            if self.pipeline is not None:
                # Move to CPU and clear cache
                self.pipeline.cpu()
                del self.pipeline
                self.pipeline = None

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("TRELLIS model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading TRELLIS model: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text-to-mesh generation using TRELLIS.

        Args:
            inputs: Dictionary containing:
                - text_prompt: Text description (required)
                - texture_text_prompt: Text description for texture generation
                - quality: Generation quality ("low",  "high")
                - texture_resolution: Texture resolution
                - output_format: Output format
                - seed: Random seed for reproducibility
                - mesh_path: Optional input path of the mesh

        Returns:
            Dictionary with generated mesh information
        """
        try:
            # Validate inputs using parent class
            output_format = self._validate_common_inputs(inputs)

            # Extract parameters
            text_prompt = inputs["text_prompt"]
            mesh_path = inputs.get("mesh_path", "")
            texture_text_prompt = inputs.get("texture_text_prompt", "")
            seed = inputs.get("seed", 42)
            texture_resolution = inputs.get("texture_resolution", 1024)
            # Postprocessing-related parameters
            simplify = inputs.get("simplify", 0.95)
            texture_bake_mode = inputs.get("texture_bake_mode", "fast")

            logger.info(f"Generating mesh with TRELLIS for prompt: '{text_prompt}'")

            # Set random seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Different Conditions
            if mesh_path:
                input_mesh = trimesh.load(mesh_path, force="mesh")
                logger.info(f"Loaded input mesh from {mesh_path}")
                outputs = self.pipeline.run_variant(
                    input_mesh,
                    prompt=text_prompt,
                    slat_sampler_params={"steps": 12, "cfg_strength": 3},
                    formats=["gaussian"],
                )
                # get ready for later texturing
                mesh = input_mesh
            else:
                # Generate 3D representation
                outputs = self.pipeline.run(
                    text_prompt,
                    texture_prompt=texture_text_prompt,
                    sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": 12, "cfg_strength": 3},
                    seed=seed,
                    formats=["gaussian", "mesh"],
                )
                mesh = None

            # Extract mesh from Gaussian representation
            mesh = self.postprocessing_utils.to_trimesh(
                outputs["gaussian"][0],
                mesh or outputs["mesh"][0],
                simplify=simplify,
                texture_size=texture_resolution,
                texture_bake_mode=texture_bake_mode,
                forward_rot=False,
            )

            # Save mesh in requested format
            output_path = self._generate_output_path(text_prompt, output_format)
            self.mesh_processor.save_mesh(mesh, output_path, do_normalise=True)

            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail_path(output_path)
            thumbnail_generated = generate_mesh_thumbnail(
                str(output_path), str(thumbnail_path)
            )

            # Create response
            response = self._create_common_response(inputs, output_format)
            response.update(
                {
                    "output_mesh_path": str(output_path),
                    "thumbnail_path": str(thumbnail_path)
                    if thumbnail_generated
                    else None,
                    "generation_info": {
                        "model": "TRELLIS",
                        "text_prompt": text_prompt,
                        "texture_prompt": texture_text_prompt,
                        "seed": seed,
                        "num_inference_steps": 12,
                        "guidance_scale": 7.5,
                        "vertex_count": len(mesh.vertices),
                        "face_count": len(mesh.faces),
                        "texture_resolution": texture_resolution,
                        "texture_bake_mode": texture_bake_mode,
                        "simplify_ratio": simplify,
                        "thumbnail_generated": thumbnail_generated,
                    },
                }
            )

            logger.info(f"TRELLIS mesh generation completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"TRELLIS mesh generation failed: {str(e)}")
            raise Exception(f"TRELLIS mesh generation failed: {str(e)}")

    def _generate_output_path(self, prompt: str, output_format: str) -> Path:
        """Generate output file path based on prompt and format."""
        # Create safe filename from prompt
        safe_name = "".join(
            c for c in prompt[:50] if c.isalnum() or c in (" ", "_")
        ).strip()
        safe_name = safe_name.replace(" ", "_")

        # Create output directory if it doesn't exist
        output_dir = Path(os.getcwd()) / "outputs" / "meshes"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        import time

        timestamp = int(time.time())
        filename = f"trellis_{safe_name}_{timestamp}.{output_format}"

        return output_dir / filename

    def _generate_thumbnail_path(self, mesh_path: Path) -> Path:
        """Generate thumbnail file path based on mesh path."""
        # Create thumbnails directory
        thumbnail_dir = Path(os.getcwd()) / "outputs" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        # Generate thumbnail filename
        thumbnail_name = mesh_path.stem + "_thumb.png"
        return thumbnail_dir / thumbnail_name

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for TRELLIS."""
        return {"input": ["text"], "output": ["glb", "obj"]}


class TrellisImageToMeshAdapterCommon(ImageToMeshModel):
    """
    Adapter for TRELLIS image-to-mesh model.

    Integrates the TRELLIS model from the thirdparty/TRELLIS directory
    into our standardized mesh generation framework.
    """

    FEATURE_TYPE = "image_to_textured_mesh"
    MODEL_ID = "trellis_image_to_textured_mesh"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vram_requirement: int = 11776,  # 12GB VRAM
        trellis_root: Optional[str] = None,
    ):
        # Set default paths
        if model_path is None:
            model_path = os.path.join(os.getcwd(), "pretrained", "TRELLIS")

        if trellis_root is None:
            trellis_root = os.path.join(os.getcwd(), "thirdparty", "TRELLIS")

        super().__init__(
            model_id=self.MODEL_ID,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_output_formats=["glb", "obj"],
            feature_type=self.FEATURE_TYPE,
        )

        self.trellis_root = Path(trellis_root)
        self.model_path = Path(model_path)
        # skip some models conditionally to save VRAM (overwrite by subclass adapaters)
        self.skip_models = ["slat_decoder_rf"]
        self.pipeline = None
        self.mesh_processor = MeshProcessor()
        # Add TRELLIS to Python path if not already there
        if str(self.trellis_root) not in sys.path:
            sys.path.insert(0, str(self.trellis_root))

    def _load_model(self):
        """Load the TRELLIS model pipeline."""
        try:
            logger.info(f"Loading TRELLIS model from {self.trellis_root}")

            # Import TRELLIS modules
            from trellis.pipelines import TrellisImageTo3DPipeline
            from trellis.utils import postprocessing_utils

            # Initialize the pipeline
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-image-large",
                cache_dir=str(self.model_path / "TRELLIS-image-large"),
                skip_models=self.skip_models,
            )
            self.pipeline.cuda()

            # Store utility modules for later use
            self.postprocessing_utils = postprocessing_utils

            logger.info("TRELLIS model loaded successfully")
            return self.pipeline

        except Exception as e:
            logger.error(f"Failed to load TRELLIS model: {str(e)}")
            raise Exception(f"Failed to load TRELLIS model: {str(e)}")

    def _unload_model(self):
        """Unload the TRELLIS model."""
        try:
            if self.pipeline is not None:
                # Move to CPU and clear cache
                self.pipeline.cpu()
                del self.pipeline
                self.pipeline = None

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("TRELLIS model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading TRELLIS model: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-mesh generation using TRELLIS.

        Args:
            inputs: Dictionary containing:
                - image_path: Text description (required)
                - texture_resolution: Texture resolution
                - output_format: Output format
                - seed: Random seed for reproducibility

        Returns:
            Dictionary with generated mesh information
        """
        try:
            if self.pipeline is None:
                raise ValueError("TRELLIS model is not loaded")

            # Validate inputs using parent class
            output_format = self._validate_common_inputs(inputs)

            # Extract parameters
            image_path = inputs["image_path"]
            seed = inputs.get("seed", 42)
            texture_resolution = inputs.get("texture_resolution", 1024)
            mesh_path = inputs.get("mesh_path", None)
            simplify = inputs.get("simplify", 0.95)
            tex_bake_mode = inputs.get("texture_bake_mode", "fast")

            logger.info(f"Generating mesh with TRELLIS for image path: '{image_path}'")

            # Set random seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Generate 3D representation
            if mesh_path:
                input_mesh = trimesh.load(mesh_path, force="mesh")
                logger.info(f"Loaded input mesh from {mesh_path}")
                # do the voxelization
                outputs = self.pipeline.run_detail_variation(
                    input_mesh,
                    Image.open(image_path),
                    seed=seed,
                    slat_sampler_params={"steps": 12, "cfg_strength": 3},
                    formats=["gaussian"],
                )
                # get ready for later texturing
                mesh = input_mesh
            else:
                outputs = self.pipeline.run(
                    Image.open(image_path),
                    preprocess_image=True,
                    sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": 12, "cfg_strength": 3},
                    seed=seed,
                    formats=["gaussian", "mesh"],
                )
                mesh = None

            # Extract mesh from Gaussian representation
            mesh = self.postprocessing_utils.to_trimesh(
                outputs["gaussian"][0],
                mesh or outputs["mesh"][0],
                simplify=simplify,
                texture_size=texture_resolution,
                texture_bake_mode=tex_bake_mode,
                forward_rot=False,
            )

            # Save mesh in requested format
            output_path = self._generate_output_path(
                image_path, output_format, is_prompt=False
            )
            self.mesh_processor.save_mesh(mesh, output_path, do_normalise=True)

            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail_path(output_path)
            thumbnail_generated = generate_mesh_thumbnail(
                str(output_path), str(thumbnail_path)
            )

            # Create response
            response = self._create_common_response(inputs, output_format)
            response.update(
                {
                    "output_mesh_path": str(output_path),
                    "thumbnail_path": str(thumbnail_path)
                    if thumbnail_generated
                    else None,
                    "generation_info": {
                        "model": "TRELLIS",
                        "image_path": image_path,
                        "seed": seed,
                        "vertex_count": len(mesh.vertices),
                        "face_count": len(mesh.faces),
                        "thumbnail_generated": thumbnail_generated,
                    },
                }
            )

            logger.info(f"TRELLIS mesh generation completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.status = ModelStatus.ERROR
            logger.error(f"TRELLIS mesh generation failed: {str(e)}")
            raise Exception(f"TRELLIS mesh generation failed: {str(e)}")

    def _generate_output_path(
        self, prompt: str, output_format: str, is_prompt: bool = True
    ) -> Path:
        """Generate output file path based on prompt and format."""
        # Create safe filename from prompt
        if is_prompt:
            safe_name = "".join(
                c for c in prompt[:50] if c.isalnum() or c in (" ", "_")
            ).strip()
            safe_name = safe_name.replace(" ", "_")
        else:
            safe_name = Path(prompt).stem[
                :50
            ]  # Use filename stem for non-prompt inputs

        # Create output directory if it doesn't exist
        output_dir = Path(os.getcwd()) / "outputs" / "meshes"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        import time

        timestamp = int(time.time())
        filename = f"trellis_{safe_name}_{timestamp}.{output_format}"

        return output_dir / filename

    def _generate_thumbnail_path(self, mesh_path: Path) -> Path:
        """Generate thumbnail file path based on mesh path."""
        # Create thumbnails directory
        thumbnail_dir = Path(os.getcwd()) / "outputs" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        # Generate thumbnail filename
        thumbnail_name = mesh_path.stem + "_thumb.png"
        return thumbnail_dir / thumbnail_name

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for TRELLIS."""
        return {"input": ["str"], "output": ["glb", "obj"]}


class TrellisTextToTexturedMeshAdapter(TrellisTextToMeshAdapterCommon):
    """
    Adapter for TRELLIS text-to-raw-mesh model.

    This adapter uses the TRELLIS model to generate raw meshes from text prompts.
    """

    FEATURE_TYPE = "text_to_textured_mesh"  # Feature type for this adapter
    MODEL_ID = "trellis_text_to_textured_mesh"

    def __init__(self, *args, **kwargs):
        kwargs["vram_requirement"] = 11776  # 12GB VRAM
        super().__init__(*args, **kwargs)
        self.supported_output_formats = ["obj", "glb"]
        self.skip_models = [
            "slat_decoder_rf"
        ]  # Skip some models conditionally to save VRAM


class TrellisTextMeshPaintingAdapter(TrellisTextToMeshAdapterCommon):
    """
    Adapter for TRELLIS text-conditioned mesh painting model.

    This adapter uses the TRELLIS model to texture meshes from text prompts.
    """

    FEATURE_TYPE = "text_mesh_painting"  # Feature type for this adapter
    MODEL_ID = "trellis_text_mesh_painting"

    def __init__(self, *args, **kwargs):
        kwargs["vram_requirement"] = 11776  # 12GB VRAM
        super().__init__(*args, **kwargs)
        self.supported_output_formats = ["obj", "glb"]
        self.skip_models = [
            "sparse_structure_decoder",
            "sparse_structure_flow_model",
            "slat_decoder_mesh",
        ]


class TrellisImageToTexturedMeshAdapter(TrellisImageToMeshAdapterCommon):
    """
    Adapter for TRELLIS text-to-textured-mesh model.

    This adapter uses the TRELLIS model to generate textured meshes from input images
    """

    FEATURE_TYPE = "image_to_textured_mesh"
    MODEL_ID = "trellis_image_to_textured_mesh"

    def __init__(self, *args, **kwargs):
        kwargs["vram_requirement"] = 11776  # 12GB VRAM
        super().__init__(*args, **kwargs)
        self.skip_models = ["slat_decoder_rf"]


class TrellisImageMeshPaintingAdapter(TrellisImageToMeshAdapterCommon):
    """
    Adapter for TRELLIS image conditioned mesh painting model.

    This adapter uses the TRELLIS model to conditionally paint meshes based on input images.
    """

    FEATURE_TYPE = "image_mesh_painting"
    MODEL_ID = "trellis_image_mesh_painting"

    def __init__(self, *args, **kwargs):
        kwargs["vram_requirement"] = 12288  # 12GB VRAM
        super().__init__(*args, **kwargs)
        self.skip_models = [
            "sparse_structure_decoder",
            "sparse_structure_flow_model",
            "slat_decoder_mesh",
        ]
