"""
HoloPart model adapter for part completion.

This adapter integrates HoloPart for 3D part completion and generation.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import trimesh

from core.models.base import ModelStatus
from core.models.mesh_models import BaseMeshGenerationModel
from utils.file_utils import OutputPathGenerator
from utils.holopart_utils import HoloPartRunner
from utils.mesh_utils import MeshProcessor

logger = logging.getLogger(__name__)


class HoloPartCompletionAdapter(BaseMeshGenerationModel):
    """
    Adapter for HoloPart part completion model.

    Integrates HoloPart for 3D part completion and generation from partial/segmented meshes.
    """

    def __init__(
        self,
        model_id: str = "holopart_part_completion",
        model_path: Optional[str] = None,
        vram_requirement: int = 10240,  # 10GB VRAM
        holopart_root: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 8,
        dtype: torch.dtype = torch.float16,
    ):
        if model_path is None:
            model_path = "pretrained/HoloPart"

        if holopart_root is None:
            holopart_root = "thirdparty/HoloPart"

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type="part_completion",
            supported_output_formats=["glb", "obj"],
        )

        self.holopart_root = Path(holopart_root)
        self.device = device
        self.batch_size = batch_size
        self.dtype = dtype

        self.holopart_runner: Optional[HoloPartRunner] = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

    def _load_model(self):
        """Load HoloPart model."""
        try:
            logger.info(f"Loading HoloPart model from {self.holopart_root}")

            # Add HoloPart to Python path
            if str(self.holopart_root) not in sys.path:
                sys.path.insert(0, str(self.holopart_root))

            # Initialize HoloPart runner
            self.holopart_runner = HoloPartRunner(
                device=self.device,
                batch_size=self.batch_size,
                holopart_weights_dir=str(self.model_path),
                dtype=self.dtype,
                holopart_root=str(self.holopart_root),
            )

            logger.info("HoloPart model loaded successfully")
            return self.holopart_runner

        except Exception as e:
            logger.error(f"Failed to load HoloPart model: {str(e)}")
            raise Exception(f"Failed to load HoloPart model: {str(e)}")

    def _unload_model(self):
        """Unload HoloPart model."""
        try:
            if self.holopart_runner is not None:
                # Clean up pipeline
                if hasattr(self.holopart_runner, "pipe") and self.holopart_runner.pipe:
                    self.holopart_runner.pipe = None
                self.holopart_runner = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("HoloPart model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading HoloPart model: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process part completion request using HoloPart.

        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input segmented mesh (GLB format with parts) (required)
                - num_inference_steps: Number of diffusion steps (default: 50)
                - guidance_scale: Guidance scale for generation (default: 3.5)
                - seed: Random seed for reproducibility (default: 2025)
                - use_flash_decoder: Whether to use flash decoder (default: True)
                - simplify_output: Whether to simplify output meshes (default: True)
                - output_format: Output format (default: "glb")

        Returns:
            Dictionary with part completion results
        """
        try:
            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for part completion")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            if not mesh_path.suffix.lower() == ".glb":
                raise ValueError("HoloPart requires GLB format with segmented parts")

            # Extract parameters
            num_inference_steps = inputs.get("num_inference_steps", 50)
            guidance_scale = inputs.get("guidance_scale", 3.5)
            seed = inputs.get("seed", 2025)
            use_flash_decoder = inputs.get("use_flash_decoder", True)
            simplify_output = inputs.get("simplify_output", True)
            output_format = inputs.get("output_format", "glb")

            # Validate output format
            if output_format not in self.supported_output_formats:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(f"Processing part completion with HoloPart: {mesh_path}")

            # Load and validate input mesh
            input_mesh = trimesh.load(mesh_path)
            if not isinstance(input_mesh, trimesh.Scene):
                raise ValueError("Input mesh must be a GLB scene with segmented parts")

            if len(input_mesh.geometry) == 0:
                raise ValueError("Input mesh scene contains no geometry")

            logger.info(f"Input mesh has {len(input_mesh.geometry)} parts")

            completed_scene = self.holopart_runner.run_holopart(
                str(mesh_path),
                num_inference_steps,
                guidance_scale,
                seed,
            )

            if completed_scene is None or len(completed_scene.geometry) == 0:
                raise Exception("HoloPart failed to generate completed parts")

            # Generate output paths
            base_name = mesh_path.stem
            output_path = self.path_generator.generate_completion_path(
                self.model_id, base_name, output_format
            )
            info_path = self.path_generator.generate_info_path(output_path)

            # Post-process completed scene
            if simplify_output:
                completed_scene = self._simplify_scene(completed_scene)

            # Save completed mesh scene
            self.mesh_processor.save_scene(completed_scene, output_path)

            # Compute statistics
            input_stats = self._compute_scene_stats(input_mesh)
            output_stats = self._compute_scene_stats(completed_scene)

            # Create completion info
            completion_info = {
                "input_parts": len(input_mesh.geometry),
                "output_parts": len(completed_scene.geometry),
                "input_stats": input_stats,
                "output_stats": output_stats,
                "generation_parameters": {
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "use_flash_decoder": use_flash_decoder,
                    "simplify_output": simplify_output,
                },
                "model_info": {
                    "model_id": self.model_id,
                    "device": self.device,
                    "batch_size": self.batch_size,
                    "dtype": str(self.dtype),
                },
            }

            # Save completion info
            self.mesh_processor.export_segmentation_info(completion_info, info_path)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "completion_info_path": str(info_path),
                "input_parts": len(input_mesh.geometry),
                "output_parts": len(completed_scene.geometry),
                "completion_info": completion_info,
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "input_format": "glb",
                    "output_format": output_format,
                    "processing_time": None,  # Could be added with timing
                },
            }

            logger.info(f"HoloPart completion completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.status = ModelStatus.ERROR
            logger.error(f"HoloPart completion failed: {str(e)}")
            raise Exception(f"HoloPart completion failed: {str(e)}")

    def _simplify_scene(self, scene: trimesh.Scene) -> trimesh.Scene:
        """Simplify all meshes in the scene."""
        simplified_scene = trimesh.Scene()

        for name, geometry in scene.geometry.items():
            try:
                if isinstance(geometry, trimesh.Trimesh):
                    simplified_geom = self.mesh_processor.simplify_mesh(geometry, 10000)
                    simplified_scene.add_geometry(simplified_geom, node_name=name)
                else:
                    # Keep non-mesh geometry as is
                    simplified_scene.add_geometry(geometry, node_name=name)
            except Exception as e:
                logger.warning(f"Failed to simplify geometry {name}: {str(e)}")
                simplified_scene.add_geometry(geometry, node_name=name)

        return simplified_scene

    def _compute_scene_stats(self, scene: trimesh.Scene) -> Dict[str, Any]:
        """Compute statistics for a mesh scene."""
        total_vertices = 0
        total_faces = 0
        part_stats = []

        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                stats = self.mesh_processor.get_mesh_stats(geometry)
                part_stats.append({"name": name, **stats})
                total_vertices += stats["vertex_count"]
                total_faces += stats["face_count"]

        return {
            "total_parts": len(scene.geometry),
            "total_vertices": total_vertices,
            "total_faces": total_faces,
            "part_stats": part_stats,
            "scene_bounds": scene.bounds.tolist() if hasattr(scene, "bounds") else None,
        }

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for HoloPart."""
        return {
            "input": ["glb"],  # HoloPart requires GLB with segmented parts
            "output": self.supported_output_formats,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = self.get_info()
        info.update(
            {
                "model_type": "part_completion",
                "description": "HoloPart model for 3D part completion and generation",
                "capabilities": [
                    "Part completion",
                    "Part generation",
                    "Diffusion-based generation",
                ],
                "requirements": [
                    "Input must be GLB format with segmented parts",
                    "Minimum 8GB VRAM recommended",
                    "CUDA support required",
                ],
            }
        )
        return info
