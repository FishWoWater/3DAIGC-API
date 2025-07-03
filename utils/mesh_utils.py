"""
Mesh processing utilities for adapters.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import trimesh

logger = logging.getLogger(__name__)


class MeshProcessor:
    """Utility class for common mesh processing operations."""

    @staticmethod
    def load_mesh(mesh_path: Union[str, Path]) -> trimesh.Trimesh:
        """Load a mesh from file."""
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        try:
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, dump it as a geometry
                mesh = mesh.to_geometry()

            logger.info(
                f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
            )
            return mesh

        except Exception as e:
            logger.error(f"Failed to load mesh from {mesh_path}: {str(e)}")
            raise

    @staticmethod
    def save_mesh(
        mesh: trimesh.Trimesh, output_path: Union[str, Path], format: str = "glb"
    ) -> Path:
        """Save a mesh to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            mesh.export(output_path)
            logger.info(f"Saved mesh to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save mesh to {output_path}: {str(e)}")
            raise

    @staticmethod
    def save_scene(
        scene: trimesh.Scene, output_path: Union[str, Path], format: str = "glb"
    ) -> Path:
        """Save a scene to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            scene.export(output_path)
            logger.info(
                f"Saved scene with {len(scene.geometry)} parts to {output_path}"
            )
            return output_path

        except Exception as e:
            logger.error(f"Failed to save scene to {output_path}: {str(e)}")
            raise

    @staticmethod
    def validate_mesh(mesh: trimesh.Trimesh) -> bool:
        """Validate mesh quality and topology."""
        try:
            # Check for empty mesh
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                logger.warning("Mesh is empty")
                return False

            # Check for degenerate faces
            if not mesh.is_valid:
                logger.warning("Mesh has invalid topology")
                return False

            # Check for manifold
            if not mesh.is_watertight:
                logger.warning("Mesh is not watertight")

            # Check for reasonable size
            if mesh.bounding_box.extents.max() < 1e-6:
                logger.warning("Mesh is too small")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating mesh: {str(e)}")
            return False

    @staticmethod
    def get_mesh_stats(mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Get basic statistics about a mesh."""
        return {
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "bounding_box": mesh.bounds.tolist(),
            "extents": mesh.bounding_box.extents.tolist(),
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "surface_area": float(mesh.area),
            "is_watertight": bool(mesh.is_watertight),
            # "is_valid": bool(mesh.is_valid),
        }

    @staticmethod
    def create_part_colors(num_parts: int) -> List[List[float]]:
        """Generate distinct colors for mesh parts."""
        import colorsys

        colors = []
        for i in range(num_parts):
            hue = i / num_parts
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append([rgb[0], rgb[1], rgb[2], 1.0])  # RGBA

        return colors

    @staticmethod
    def export_segmentation_info(
        segmentation_data: Dict[str, Any], output_path: Union[str, Path]
    ) -> Path:
        """Export segmentation information to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w") as f:
                json.dump(segmentation_data, f, indent=2, default=str)

            logger.info(f"Saved segmentation info to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save segmentation info: {str(e)}")
            raise

    @staticmethod
    def export_generation_info(
        generation_data: Dict[str, Any], output_path: Union[str, Path]
    ) -> Path:
        """Export generation information to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w") as f:
                json.dump(generation_data, f, indent=2, default=str)

            logger.info(f"Saved generation info to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save generation info: {str(e)}")
            raise

    @staticmethod
    def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        """Simplify mesh to target number of faces."""
        try:
            if len(mesh.faces) <= target_faces:
                return mesh

            simplified = mesh.simplify_quadric_decimation(target_faces)
            logger.info(
                f"Simplified mesh from {len(mesh.faces)} to {len(simplified.faces)} faces"
            )
            return simplified

        except Exception as e:
            logger.warning(f"Failed to simplify mesh: {str(e)}")
            return mesh

    @staticmethod
    def normalize_mesh(mesh: trimesh.Trimesh, scale: float = 1.0) -> trimesh.Trimesh:
        """Normalize mesh to fit within a unit sphere."""
        try:
            # Center the mesh
            mesh.apply_translation(-mesh.centroid)

            # Scale to fit within sphere
            max_extent = mesh.bounding_box.extents.max()
            if max_extent > 0:
                mesh.apply_scale(scale / max_extent)

            return mesh

        except Exception as e:
            logger.error(f"Failed to normalize mesh: {str(e)}")
            return mesh
