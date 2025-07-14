"""
PartPacker model adapter for part-level 3D mesh generation from single-view images.

This adapter integrates PartPacker for generating 3D meshes with part-level decomposition
from single-view images.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from core.models.base import ModelStatus
from core.models.mesh_models import ImageToMeshModel
from utils.file_utils import OutputPathGenerator
from utils.mesh_utils import MeshProcessor
from utils.partpacker_utils import PartPackerRunner

logger = logging.getLogger(__name__)


class PartPackerImageToRawMeshAdapter(ImageToMeshModel):
    """
    Adapter for PartPacker part-level mesh generation from single-view images.

    Integrates PartPacker for generating 3D meshes with semantic part decomposition.
    """

    def __init__(
        self,
        model_id: str = "partpacker_image_to_raw_mesh",
        model_path: Optional[str] = None,
        vram_requirement: int = 10240,  # 10GB VRAM
        partpacker_root: Optional[str] = None,
    ):
        if model_path is None:
            model_path = "pretrained/PartPacker/flow.pt"

        if partpacker_root is None:
            partpacker_root = "thirdparty/PartPacker"

        super().__init__(
            model_id=model_id, model_path=model_path, vram_requirement=vram_requirement
        )

        self.partpacker_root = Path(partpacker_root)
        self.partpacker_runner = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

        # Configuration paths
        self.flow_ckpt_path = model_path

    def _load_model(self):
        """Load PartPacker model."""
        try:
            logger.info(f"Loading PartPacker model from {self.partpacker_root}")

            # Add PartPacker to Python path
            if str(self.partpacker_root) not in sys.path:
                sys.path.insert(0, str(self.partpacker_root))

            # Initialize PartPacker runner
            self.partpacker_runner = PartPackerRunner(
                config_name="default",
                flow_ckpt_path=self.flow_ckpt_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision="bfloat16",
                # TODO: move all remove background to some common utilities
                enable_background_removal=True,
                partpacker_root=str(self.partpacker_root),
            )

            logger.info("PartPacker model loaded successfully")
            return self.partpacker_runner

        except Exception as e:
            logger.error(f"Failed to load PartPacker model: {str(e)}")
            raise Exception(f"Failed to load PartPacker model: {str(e)}")

    def _unload_model(self):
        """Unload PartPacker model."""
        try:
            if self.partpacker_runner is not None:
                self.partpacker_runner.cleanup()
                self.partpacker_runner = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("PartPacker model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading PartPacker model: {str(e)}")

    def _generate_thumbnail_path(self, mesh_path: Path) -> Path:
        """Generate thumbnail file path based on mesh path."""
        import os

        # Create thumbnails directory
        thumbnail_dir = Path(os.getcwd()) / "outputs" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        # Generate thumbnail filename
        thumbnail_name = mesh_path.stem + "_thumb.png"
        return thumbnail_dir / thumbnail_name

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-mesh generation request using PartPacker.

        Args:
            inputs: Dictionary containing:
                - image_path: Path to input image (required)
                - output_format: Output format (default: "glb")
                - num_steps: Number of diffusion steps (default: 30)
                - cfg_scale: Classifier-free guidance scale (default: 7.0)
                - grid_resolution: Grid resolution for mesh extraction (default: 384)
                - num_faces: Target number of faces (default: 50000)
                - seed: Random seed for reproducibility (optional)
                - return_parts: Whether to save individual parts (default: True)
                - return_volumes: Whether to save dual volumes (default: False)

        Returns:
            Dictionary with generation results
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
            num_steps = inputs.get("num_steps", 30)
            cfg_scale = inputs.get("cfg_scale", 7.0)
            grid_resolution = inputs.get("grid_resolution", 384)
            num_faces = inputs.get("num_faces", 50000)
            seed = inputs.get("seed", None)
            return_parts = inputs.get("return_parts", True)
            return_volumes = inputs.get("return_volumes", False)

            if output_format not in ["glb", "obj", "ply"]:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(
                f"Generating part-level mesh with PartPacker from image: {image_path}"
            )

            generation_result = self.partpacker_runner.generate_from_image(
                str(image_path),
                num_steps,
                cfg_scale,
                grid_resolution,
                num_faces,
                seed,
                return_parts,
                return_volumes,
            )

            if generation_result is None:
                raise Exception("PartPacker generation failed to produce results")

            # Generate output paths
            base_name = f"{self.model_id}_{image_path.stem}"
            output_path = self.path_generator.generate_mesh_path(
                self.model_id, base_name, output_format
            )
            info_path = self.path_generator.generate_info_path(output_path)

            # Save main combined mesh
            combined_mesh = generation_result["combined_mesh"]
            if combined_mesh is not None:
                combined_mesh.export(str(output_path))
            else:
                raise Exception("No combined mesh generated")

            # Save individual parts if requested
            part_files = []
            if return_parts and generation_result["parts"]:
                parts_dir = output_path.parent / f"{output_path.stem}_parts"
                parts_dir.mkdir(exist_ok=True)

                for i, part in enumerate(generation_result["parts"]):
                    part_path = parts_dir / f"part_{i:02d}.{output_format}"
                    part.export(str(part_path))
                    part_files.append(str(part_path))

            # Save dual volumes if requested
            volume_files = []
            if return_volumes and generation_result["dual_volumes"]:
                volumes_dir = output_path.parent / f"{output_path.stem}_volumes"
                volumes_dir.mkdir(exist_ok=True)

                for i, volume in enumerate(generation_result["dual_volumes"]):
                    volume_path = volumes_dir / f"volume_{i:02d}.{output_format}"
                    volume.export(str(volume_path))
                    volume_files.append(str(volume_path))

            # Get mesh statistics
            final_mesh = self.mesh_processor.load_mesh(output_path)
            mesh_stats = self.mesh_processor.get_mesh_stats(final_mesh)

            # Create generation info
            generation_info = {
                "num_parts": len(generation_result["parts"])
                if generation_result["parts"]
                else 1,
                "num_volumes": len(generation_result["dual_volumes"])
                if generation_result["dual_volumes"]
                else 0,
                "generation_parameters": generation_result["metadata"],
                "mesh_stats": mesh_stats,
                "part_files": part_files,
                "volume_files": volume_files,
                "model_info": self.partpacker_runner.get_model_info(),
            }

            # Save generation info
            self.mesh_processor.export_generation_info(generation_info, info_path)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "generation_info_path": str(info_path),
                "num_parts": generation_info["num_parts"],
                "num_volumes": generation_info["num_volumes"],
                "part_files": part_files,
                "volume_files": volume_files,
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_image": str(image_path),
                    "output_format": output_format,
                    "vertex_count": mesh_stats["vertex_count"],
                    "face_count": mesh_stats["face_count"],
                    "num_steps": num_steps,
                    "cfg_scale": cfg_scale,
                    "grid_resolution": grid_resolution,
                    "seed": seed,
                    "has_parts": len(generation_result["parts"]) > 1
                    if generation_result["parts"]
                    else False,
                },
            }

            logger.info(f"PartPacker generation completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"PartPacker generation failed: {str(e)}")
            raise Exception(f"PartPacker generation failed: {str(e)}")

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for PartPacker."""
        return {"input": ["png", "jpg", "jpeg"], "output": ["glb", "obj"]}
