"""
UniRig model adapter for automatic rigging of 3D meshes.

This adapter integrates the UniRig fast inference engine for automatic
mesh rigging with skeleton generation and skin weight computation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from core.models.base import ModelStatus
from core.models.rig_models import AutoRigModel
from utils.file_utils import OutputPathGenerator
from utils.mesh_utils import MeshProcessor
from utils.unirig_utils import InferenceConfig, UniRigInferenceEngine

logger = logging.getLogger(__name__)


class UniRigAdapter(AutoRigModel):
    """
    Adapter for UniRig automatic rigging model.

    Integrates UniRig's fast inference engine for automatic mesh rigging
    with skeleton generation and skin weight computation.
    """

    def __init__(
        self,
        model_id: str = "unirig_auto_rig",
        model_path: Optional[str] = None,
        vram_requirement: int = 9216,  # 9GB VRAM
        unirig_root: Optional[str] = None,
        device: str = "cuda",
    ):
        if model_path is None:
            model_path = "pretrained/UniRig"

        if unirig_root is None:
            unirig_root = "thirdparty/UniRig"

        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            supported_input_formats=["fbx", "obj", "glb"],
            supported_output_formats=["fbx", "glb"],
        )

        self.unirig_root = Path(unirig_root)
        self.device = device
        self.inference_engine: Optional[UniRigInferenceEngine] = None
        self.mesh_processor = MeshProcessor()
        self.path_generator = OutputPathGenerator(base_output_dir="outputs")

        # Verify UniRig installation
        if not self.unirig_root.exists():
            raise FileNotFoundError(f"UniRig not found at: {self.unirig_root}")

    def _load_model(self):
        """Load UniRig inference engine."""
        try:
            logger.info(f"Loading UniRig model from {self.unirig_root}")

            # Add UniRig to Python path
            if str(self.unirig_root) not in sys.path:
                sys.path.insert(0, str(self.unirig_root))

            # Create inference configuration
            config = InferenceConfig(
                device=self.device,
                compile_model=True,  # Use torch.compile for faster inference
                cache_dir=str(
                    self.path_generator.base_output_dir / "temp" / "unirig_cache"
                ),
                precision="bf16-mixed",
            )

            # Initialize inference engine
            self.inference_engine = UniRigInferenceEngine(config)

            # Preload models for faster inference
            self.inference_engine.preload_systems()

            logger.info("UniRig model loaded successfully")
            return self.inference_engine

        except Exception as e:
            logger.error(f"Failed to load UniRig model: {str(e)}")
            raise Exception(f"Failed to load UniRig model: {str(e)}")

    def _unload_model(self):
        """Unload UniRig model."""
        try:
            if self.inference_engine is not None:
                # Clear model cache
                self.inference_engine.clear_cache()
                self.inference_engine = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("UniRig model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading UniRig model: {str(e)}")

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process auto-rigging request using UniRig.

        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh (required)
                - rig_mode: Rigging mode ("skeleton", "skin", "full") (default: "full")
                - output_format: Output format ("fbx", "obj", "glb") (default: "fbx")
                - seed: Random seed for generation (default: None)
                - with_skinning: Whether to apply skinning weights (default: True)
                - skeleton_config: Path to skeleton task config (default: None)
                - skin_config: Path to skin task config (default: None)

        Returns:
            Dictionary with rigging results
        """
        try:
            if self.inference_engine is None:
                raise ValueError("UniRig inference engine is not loaded")

            # Validate inputs
            if "mesh_path" not in inputs:
                raise ValueError("mesh_path is required for auto-rigging")

            mesh_path = Path(inputs["mesh_path"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

            # Extract parameters
            rig_mode = inputs.get("rig_mode", "full")
            output_format = inputs.get("output_format", "fbx")
            seed = inputs.get("seed", None)
            with_skinning = inputs.get("with_skinning", True)
            skeleton_config = inputs.get("skeleton_config", None)
            skin_config = inputs.get("skin_config", None)

            # Validate rig mode
            if rig_mode not in ["skeleton", "skin", "full"]:
                raise ValueError(
                    f"Invalid rig_mode: {rig_mode}. Must be 'skeleton', 'skin', or 'full'"
                )

            logger.info(f"Auto-rigging mesh with UniRig: {mesh_path}, mode: {rig_mode}")

            # Load and validate mesh
            try:
                mesh = self.mesh_processor.load_mesh(mesh_path)
                # if not self.mesh_processor.validate_mesh(mesh):
                # logger.warning("Input mesh validation failed, proceeding anyway")
                mesh_stats = self.mesh_processor.get_mesh_stats(mesh)
            except Exception as e:
                logger.warning(f"Failed to analyze input mesh: {e}")
                mesh_stats = {"vertex_count": 0, "face_count": 0}

            # Generate output paths
            base_name = mesh_path.stem
            output_path = self.path_generator.generate_rigged_path(
                self.model_id, base_name, output_format
            )
            output_dir = output_path.parent

            output_filename = output_path.name

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            if rig_mode == "skeleton":
                # NOTE: using await here will make blender context incorrect
                result_path = self.inference_engine.generate_skeleton(
                    str(mesh_path),
                    str(output_dir),
                    os.path.join(
                        str(output_dir), output_filename
                    ),  # notice that this should ACTUALLY be output_path
                    skeleton_config,
                )
                has_skinning = False

            elif rig_mode == "skin":
                result_path = self.inference_engine.generate_skin_weights(
                    str(mesh_path),
                    str(output_dir),
                    os.path.join(str(output_dir), output_filename),
                    skin_config,
                )
                has_skinning = True

            else:  # full pipeline
                result_path = self.inference_engine.full_pipeline(
                    str(mesh_path),
                    str(output_dir),
                    os.path.join(str(output_dir), output_filename),
                )
                has_skinning = with_skinning

            # Verify output was created
            if not Path(result_path).exists():
                raise Exception(f"UniRig failed to generate output file: {result_path}")

            # Estimate bone count from output (simplified approach)
            bone_count = self._estimate_bone_count(Path(result_path))

            # Create rig info
            rig_info = {
                "rig_type": "auto_detected",
                "has_skinning": has_skinning,
                "skeleton_only": rig_mode == "skeleton",
                "generation_method": "unirig_fast_inference",
                "bone_count": bone_count,
                "rig_mode": rig_mode,
            }

            # Create response
            response = {
                "output_mesh_path": str(result_path),
                "bone_count": bone_count,
                "rig_info": rig_info,
                "format": output_format,
                "success": True,
                "generation_info": {
                    "model": self.model_id,
                    "input_mesh": str(mesh_path),
                    "vertex_count": mesh_stats.get("vertex_count", 0),
                    "face_count": mesh_stats.get("face_count", 0),
                    "rig_mode": rig_mode,
                    "device": self.device,
                    "seed": seed,
                },
            }

            logger.info(f"UniRig auto-rigging completed: {result_path}")
            self.status = ModelStatus.LOADED
            return response

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.status = ModelStatus.ERROR
            logger.error(f"UniRig auto-rigging failed: {str(e)}")
            raise Exception(f"UniRig auto-rigging failed: {str(e)}")

    def _generate_thumbnail_path(self, mesh_path: Path) -> Path:
        """Generate thumbnail file path based on mesh path."""
        # Create thumbnails directory
        thumbnail_dir = Path(os.getcwd()) / "outputs" / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        # Generate thumbnail filename
        thumbnail_name = mesh_path.stem + "_thumb.png"
        return thumbnail_dir / thumbnail_name

    def _estimate_bone_count(self, rigged_file: Path) -> int:
        """
        Estimate bone count from rigged file.

        This is a simplified implementation. In practice, you would
        parse the file format to count actual bones.
        """
        try:
            # For FBX files, we could parse and count bones
            # For now, return a reasonable estimate based on file size
            file_size = rigged_file.stat().st_size

            # Rough heuristic: larger files typically have more bones
            if file_size > 10 * 1024 * 1024:  # > 10MB
                return 50  # Complex rig
            elif file_size > 5 * 1024 * 1024:  # > 5MB
                return 30  # Medium rig
            elif file_size > 1 * 1024 * 1024:  # > 1MB
                return 20  # Simple rig
            else:
                return 15  # Minimal rig

        except Exception as e:
            logger.warning(f"Failed to estimate bone count: {e}")
            return 20  # Default estimate

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats for UniRig."""
        return {"input": ["fbx", "obj", "glb"], "output": ["fbx", "glb"]}

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information for UniRig."""
        info = super().get_model_info()
        info.update(
            {
                "model_name": "UniRig",
                "version": "1.0",
                "description": "Unified automatic rigging using fast inference engine",
                "capabilities": [
                    "Automatic skeleton generation",
                    "Skin weight prediction",
                ],
                "stages": [
                    "Skeleton prediction using autoregressive transformer",
                    "Skin weight computation using bone-point cross attention",
                ],
                "requirements": {
                    "vram_gb": 8,
                    "pytorch_version": ">=2.3.1",
                    "cuda_required": True,
                },
                "interface": "fast_inference_engine",
                "supported_modes": ["skeleton", "skin", "full"],
                "performance": {
                    "skeleton_generation": "~30-60 seconds",
                    "skin_generation": "~60-120 seconds",
                    "full_pipeline": "~90-180 seconds",
                },
            }
        )
        return info
