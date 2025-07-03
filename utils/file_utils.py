"""
File handling utilities for adapters.
"""

import logging
import time
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class OutputPathGenerator:
    """Utility class for generating output file paths."""

    def __init__(self, base_output_dir: Union[str, Path] = "outputs"):
        self.base_output_dir = Path(base_output_dir)

    def generate_mesh_path(
        self,
        model_id: str,
        base_name: str,
        output_format: str = "glb",
        subdirectory: str = "meshes",
    ) -> Path:
        """Generate output path for mesh files."""
        output_dir = self.base_output_dir / subdirectory
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        filename = f"{model_id}_{base_name}_{timestamp}.{output_format}"

        return output_dir / filename

    def generate_segmentation_path(
        self,
        model_id: str,
        base_name: str,
        output_format: str = "glb",
        subdirectory: str = "segmented",
    ) -> Path:
        """Generate output path for segmented mesh files."""
        output_dir = self.base_output_dir / subdirectory
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        filename = f"{model_id}_{base_name}_{timestamp}.{output_format}"

        return output_dir / filename

    def generate_completion_path(
        self,
        model_id: str,
        base_name: str,
        output_format: str = "glb",
        subdirectory: str = "completed",
    ) -> Path:
        """Generate output path for part completion files."""
        output_dir = self.base_output_dir / subdirectory
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        filename = f"{model_id}_{base_name}_{timestamp}.{output_format}"

        return output_dir / filename

    def generate_rigged_path(
        self,
        model_id: str,
        base_name: str,
        output_format: str = "fbx",
        subdirectory: str = "rigged",
    ) -> Path:
        """Generate output path for rigged mesh files."""
        output_dir = self.base_output_dir / subdirectory
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        filename = f"{model_id}_{base_name}_{timestamp}.{output_format}"

        return output_dir / filename

    def generate_info_path(self, mesh_path: Path) -> Path:
        """Generate corresponding JSON info path for a mesh file."""
        return mesh_path.with_suffix(".json")

    def generate_temp_path(self, base_name: str, extension: str = "glb") -> Path:
        """Generate temporary file path."""
        temp_dir = self.base_output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        filename = f"temp_{base_name}_{timestamp}.{extension}"

        return temp_dir / filename

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files older than specified hours."""
        temp_dir = self.base_output_dir / "temp"
        if not temp_dir.exists():
            return

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        for temp_file in temp_dir.iterdir():
            if temp_file.is_file():
                file_age = current_time - temp_file.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        temp_file.unlink()
                        logger.info(f"Cleaned up temp file: {temp_file}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to cleanup temp file {temp_file}: {str(e)}"
                        )
