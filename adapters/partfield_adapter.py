"""
PartField model adapter for mesh segmentation.

This adapter integrates PartField for semantic mesh segmentation using
the PartFieldRunner from utils.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from core.models.base import ModelStatus
from core.models.segment_models import MeshSegmentationModel
from utils.file_utils import OutputPathGenerator
from utils.mesh_utils import MeshProcessor
from utils.partfield_utils import PartFieldRunner

logger = logging.getLogger(__name__)


class PartFieldSegmentationAdapter(MeshSegmentationModel):
    """
    Adapter for PartField mesh segmentation model.

    Integrates PartField for semantic mesh segmentation.
    """

    def __init__(
        self,
        model_id: str = "partfield_mesh_segmentation",
        model_path: Optional[str] = None,
        vram_requirement: int = 4096,  # 4GB VRAM
        partfield_root: Optional[str] = None,
    ):
        if model_path is None:
            model_path = "pretrained/PartField/model_objaverse.pt"

        if partfield_root is None:
            partfield_root = "thirdparty/PartField"

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
        )

        self.partfield_root = Path(partfield_root)
        self.partfield_runner: Optional[PartFieldRunner] = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(
            base_output_dir="exp_results"
        )  # some constraints on hard-coded rules of PartField, TBF

        # Configuration paths
        self.config_file = str(self.partfield_root / "configs" / "final" / "demo.yaml")
        self.continue_ckpt = model_path

    def _load_model(self):
        """Load PartField segmentation model."""
        try:
            logger.info(f"Loading PartField model from {self.partfield_root}")

            # Add PartField to Python path
            if str(self.partfield_root) not in sys.path:
                sys.path.insert(0, str(self.partfield_root))

            # Initialize PartField runner
            self.partfield_runner = PartFieldRunner(
                config_file=self.config_file,
                continue_ckpt=self.continue_ckpt,
                partfield_root=str(self.partfield_root),
            )

            logger.info("PartField model loaded successfully")
            return self.partfield_runner

        except Exception as e:
            logger.error(f"Failed to load PartField model: {str(e)}")
            raise Exception(f"Failed to load PartField model: {str(e)}")

    def _unload_model(self):
        """Unload PartField model."""
        try:
            if self.partfield_runner is not None:
                # Clean up any resources
                self.partfield_runner = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("PartField model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading PartField model: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process mesh segmentation request using PartField.

        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - num_parts: Target number of parts (default: 6)
                - use_hierarchical: Whether to use hierarchical clustering (default: True)
                - alg_option: Algorithm option (0, 1, 2) (default: 0)
                - export_colored_mesh: Whether to export colored PLY files (default: True)

        Returns:
            Dictionary with segmentation results
        """
        try:
            if self.partfield_runner is None:
                raise RuntimeError("PartField runner is not loaded")

            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for mesh segmentation")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            # Extract parameters
            num_parts = inputs.get("num_parts", 6)
            segmentation_method = inputs.get("segmentation_method", "semantic")
            use_hierarchical = inputs.get("use_hierarchical", True)
            alg_option = inputs.get("alg_option", 0)
            export_colored_mesh = inputs.get("export_colored_mesh", True)

            logger.info(f"Segmenting mesh with PartField: {mesh_path}")

            # Load and validate mesh
            mesh = self.mesh_processor.load_mesh(mesh_path)
            # if not self.mesh_processor.validate_mesh(mesh):
            # logger.warning("Input mesh validation failed, proceeding anyway")

            # Setup temporary directories for PartField
            base_name = mesh_path.stem
            temp_base = (
                self.path_generator.base_output_dir
                / "temp"
                / f"partfield_{int(time.time())}"
            )
            feature_dir = temp_base / "features"
            cluster_dir = temp_base / "clustering"

            # Create required directory structure
            feature_dir.mkdir(parents=True, exist_ok=True)
            cluster_dir.mkdir(parents=True, exist_ok=True)

            # Update runner configuration for this request

            segmentation_result, num_parts_to_path = (
                self.partfield_runner.run_partfield(
                    str(mesh_path), str(feature_dir), str(cluster_dir), num_parts
                )
            )

            if segmentation_result is None:
                raise Exception("PartField segmentation failed to produce results")

            # Generate output paths
            output_path = self.path_generator.generate_segmentation_path(
                self.model_id, base_name, "glb"
            )
            info_path = self.path_generator.generate_info_path(output_path)

            # Create segmented mesh scene
            scene = self._create_segmented_scene(mesh, segmentation_result, num_parts)

            # Save segmented mesh
            self.mesh_processor.save_scene(scene, output_path, do_normalise=True)

            # Create segmentation info
            segmentation_info = {
                "num_parts": num_parts,
                "segmentation_method": segmentation_method,
                "use_hierarchical": use_hierarchical,
                "alg_option": alg_option,
                "mesh_stats": self.mesh_processor.get_mesh_stats(mesh),
                "part_statistics": self._compute_part_statistics(
                    segmentation_result, num_parts
                ),
                "processing_parameters": {
                    "export_colored_mesh": export_colored_mesh,
                    "model_id": self.model_id,
                },
            }

            # Save segmentation info
            self.mesh_processor.export_segmentation_info(segmentation_info, info_path)

            # Cleanup temporary files
            self._cleanup_temp_files(temp_base)

            # Create response
            response = {
                "output_mesh_path": str(output_path),
                "segmentation_info_path": str(info_path),
                "num_parts": num_parts,
                "segmentation_info": segmentation_info,
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "vertex_count": len(mesh.vertices),
                    "face_count": len(mesh.faces),
                    "segmentation_method": segmentation_method,
                },
            }

            logger.info(f"PartField segmentation completed: {output_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.status = ModelStatus.ERROR
            logger.error(f"PartField segmentation failed: {str(e)}")
            raise Exception(f"PartField segmentation failed: {str(e)}")

    def _create_segmented_scene(self, mesh, segmentation_result, num_parts):
        """Create a trimesh scene with segmented parts."""
        import numpy as np
        import trimesh

        scene = trimesh.Scene()

        if isinstance(segmentation_result, np.ndarray):
            # Handle single segmentation result
            labels = segmentation_result.flatten()
        else:
            # Handle hierarchical results - use the one with desired number of segments
            if hasattr(segmentation_result, "__len__") and len(segmentation_result) > 0:
                # Find the segmentation closest to desired number of segments
                # best_idx = min(len(segmentation_result) - 1, num_parts - 1)
                ### TODO: we may return a number of paths to the client, to convineiently swicthes to different parts
                best_idx = 0
                labels = segmentation_result[best_idx]
                if isinstance(labels, np.ndarray):
                    labels = labels.flatten().tolist()
            else:
                # Fallback: create random segmentation
                logger.error(
                    "No hierarchical segmentation result found, using random segmentation"
                )
                labels = np.random.randint(0, num_parts, len(mesh.faces))

        # Create parts based on face labels
        unique_labels = np.unique(labels)
        colors = self.mesh_processor.create_part_colors(len(unique_labels))

        for i, label in enumerate(unique_labels):
            # Get faces for this part
            face_mask = labels == label
            if not np.any(face_mask):
                continue

            # Create submesh for this part
            try:
                part_mesh = mesh.submesh([face_mask], only_watertight=False)
                if isinstance(part_mesh, list) and len(part_mesh) > 0:
                    part_mesh = part_mesh[0]

                # Assign color
                if (
                    isinstance(part_mesh, trimesh.Trimesh)
                    and hasattr(part_mesh, "visual")
                    and hasattr(part_mesh.visual, "face_colors")
                ):
                    part_mesh.visual.face_colors = colors[i % len(colors)]

                scene.add_geometry(part_mesh, node_name=f"part_{label}")

            except Exception as e:
                logger.warning(f"Failed to create submesh for part {label}: {str(e)}")
                continue

        # If no parts were created, add the original mesh
        if len(scene.geometry) == 0:
            scene.add_geometry(mesh, node_name="original_mesh")

        return scene

    def _compute_part_statistics(self, segmentation_result, num_parts):
        """Compute statistics for segmented parts."""
        import numpy as np

        if isinstance(segmentation_result, np.ndarray):
            labels = segmentation_result.flatten()
            unique_labels, counts = np.unique(labels, return_counts=True)

            return {
                "num_parts_actual": len(unique_labels),
                "num_parts_requested": num_parts,
                "part_sizes": {
                    int(label): int(count)
                    for label, count in zip(unique_labels, counts)
                },
                "average_part_size": float(np.mean(counts)),
                "part_size_std": float(np.std(counts)),
            }
        else:
            return {
                "num_parts_actual": num_parts,
                "num_parts_requested": num_parts,
                "hierarchical_levels": len(segmentation_result)
                if hasattr(segmentation_result, "__len__")
                else 1,
            }

    def _cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary files created during processing."""
        try:
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(
                f"Failed to cleanup temporary directory {temp_dir}: {str(e)}"
            )

    def _generate_thumbnail_path(self, mesh_path: Path) -> Path:
        """Generate thumbnail file path based on mesh path."""
        import os

        # Create thumbnails directory
        thumbnail_dir = Path(os.getcwd()) / "outputs" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        # Generate thumbnail filename
        thumbnail_name = mesh_path.stem + "_thumb.png"
        return thumbnail_dir / thumbnail_name

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for PartField."""
        return {"input": ["glb", "obj"], "output": ["glb"]}
