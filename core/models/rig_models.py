"""
Auto-rigging models for adding armatures to 3D meshes.

This module provides models that can automatically generate rigging/armature
for 3D meshes without existing bone structures.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)


class AutoRigModel(BaseModel):
    """
    Auto-rigging model that adds bone structure to meshes.

    Inputs: A OBJ/GLB/FBX mesh without rig/armature
    Outputs: A OBJ/GLB/FBX mesh with rig/armature
    """

    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        supported_input_formats: Optional[List[str]] = None,
        supported_output_formats: Optional[List[str]] = None,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type="auto_rig",
        )

        self.supported_input_formats = supported_input_formats or ["obj", "glb", "fbx"]
        self.supported_output_formats = supported_output_formats or [
            "glb",
            "fbx",
        ]

    def _load_model(self):
        """Load the auto-rigging model. To be implemented by adapters."""
        logger.info(f"Loading auto-rig model: {self.model_id}")
        # This will be implemented by specific adapters (e.g., UniRig)
        pass

    def _unload_model(self):
        """Unload the auto-rigging model."""
        logger.info(f"Unloading auto-rig model: {self.model_id}")
        # This will be implemented by specific adapters
        pass

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process auto-rigging request.

        Args:
            inputs: Dictionary containing:
                - mesh_path: Path to input mesh file (required)
                - output_format: Desired output format (optional, defaults to input format)
                - rig_type: Type of rig to generate (optional, e.g., "humanoid", "quadruped")
                - bone_count: Target number of bones (optional)
                - symmetry: Whether to enforce symmetry (optional, default True)

        Returns:
            Dictionary containing:
                - output_mesh_path: Path to rigged mesh file
                - bone_count: Number of bones generated
                - rig_info: Information about the generated rig
        """
        # Validate inputs
        if "mesh_path" not in inputs:
            raise ValueError("mesh_path is required for auto-rigging")

        mesh_path = Path(inputs["mesh_path"])
        if not mesh_path.exists():
            raise FileNotFoundError(f"Input mesh file not found: {mesh_path}")

        # Check input format
        input_format = mesh_path.suffix.lower().lstrip(".")
        if input_format not in self.supported_input_formats:
            raise ValueError(f"Unsupported input format: {input_format}")

        # Get output format (default to input format)
        output_format = inputs.get("output_format", input_format)
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")

        logger.info(f"Processing auto-rig request for {mesh_path}")

        # This will be implemented by specific adapters
        # For now, return a placeholder response
        return {
            "output_mesh_path": str(
                mesh_path.parent / f"rigged_{mesh_path.stem}.{output_format}"
            ),
            "bone_count": inputs.get("bone_count", 20),
            "rig_info": {
                "rig_type": inputs.get("rig_type", "humanoid"),
                "symmetry": inputs.get("symmetry", True),
                "success": True,
            },
        }

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats."""
        return {
            "input": self.supported_input_formats,
            "output": self.supported_output_formats,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = self.get_info()
        info.update(
            {
                "model_type": "auto_rig",
                "description": "Automatic rigging model for adding bone structures to meshes",
                "capabilities": [
                    "skeleton generation",
                    "skin generation",
                ],
            }
        )
        return info
