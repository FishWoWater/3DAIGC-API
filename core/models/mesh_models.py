"""
Mesh generation models for creating 3D meshes from various inputs.

This module provides models that can generate textured meshes from text prompts,
images, or combinations of both inputs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)


class BaseMeshGenerationModel(BaseModel):
    """
    Base class for mesh generation models.

    Common functionality for all mesh generation models regardless of input type.
    """

    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        feature_type: str = "mesh_generation",
        supported_output_formats: Optional[List[str]] = None,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=feature_type,
        )

        self.supported_output_formats = supported_output_formats or ["glb", "obj"]

    def _validate_common_inputs(self, inputs: Dict[str, Any]) -> str:
        """Validate common inputs and return output format."""
        # Get output format
        output_format = inputs.get("output_format", "glb")
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")

        return output_format

    def _create_common_response(
        self, inputs: Dict[str, Any], output_format: str
    ) -> Dict[str, Any]:
        """Create common response structure."""
        return {
            "output_mesh_path": f"generated_mesh_{self.model_id}.{output_format}",
            "success": True,
        }


class TextToMeshModel(BaseMeshGenerationModel):
    """
    Text-conditioned mesh generation model.

    Inputs: Text prompt
    Outputs: A textured mesh
    """

    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        feature_type: str = "text_to_textured_mesh",
        supported_output_formats: Optional[List[str]] = None,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=feature_type,
            supported_output_formats=supported_output_formats,
        )

    def _load_model(self):
        """Load the text-to-mesh model. To be implemented by adapters."""
        logger.info(f"Loading text-to-mesh model: {self.model_id}")
        # This will be implemented by specific adapters (e.g., TRELLIS, Hunyuan3D)
        pass

    def _unload_model(self):
        """Unload the text-to-mesh model."""
        logger.info(f"Unloading text-to-mesh model: {self.model_id}")
        # This will be implemented by specific adapters
        pass

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text-to-mesh generation request.

        Args:
            inputs: Dictionary containing:
                - text_prompt: Text description of the mesh to generate (required)
                - texture_resolution: Texture resolution in pixels
                - output_format: Output format (glb, obj, ply)
                - seed: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - output_mesh_path: Path to generated mesh file
                - texture_resolution: Applied texture resolution
                - generation_info: Additional generation metadata
        """
        # Validate inputs
        if "text_prompt" not in inputs:
            raise ValueError("text_prompt is required for text-to-mesh generation")

        text_prompt = inputs["text_prompt"].strip()
        if not text_prompt:
            raise ValueError("text_prompt cannot be empty")

        output_format = self._validate_common_inputs(inputs)

        logger.info(f"Processing text-to-mesh request: '{text_prompt}'")

        # This will be implemented by specific adapters
        response = self._create_common_response(inputs, output_format)
        response.update(
            {
                "text_prompt": text_prompt,
                "seed": inputs.get("seed"),
                "generation_info": {
                    "input_type": "text",
                    "prompt_length": len(text_prompt),
                    "success": True,
                },
            }
        )

        return response

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats."""
        return {"input": ["text"], "output": self.supported_output_formats}


class ImageToMeshModel(BaseMeshGenerationModel):
    """
    Image-conditioned mesh generation model.

    Inputs: Single image or multiple images
    Outputs: A textured mesh
    """

    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
        supported_input_formats: Optional[List[str]] = None,
        supported_output_formats: Optional[List[str]] = None,
        feature_type: str = "image_to_textured_mesh",
        max_images: int = 4,
    ):
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            feature_type=feature_type,
            supported_output_formats=supported_output_formats,
        )

        self.supported_input_formats = supported_input_formats or [
            "jpg",
            "jpeg",
            "png",
            "webp",
        ]
        self.max_images = max_images

    def _load_model(self):
        """Load the image-to-mesh model. To be implemented by adapters."""
        logger.info(f"Loading image-to-mesh model: {self.model_id}")
        # This will be implemented by specific adapters
        pass

    def _unload_model(self):
        """Unload the image-to-mesh model."""
        logger.info(f"Unloading image-to-mesh model: {self.model_id}")
        # This will be implemented by specific adapters
        pass

    def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image-to-mesh generation request.

        Args:
            inputs: Dictionary containing:
                - image_paths: List of image file paths or single image path (required)
                - quality: Generation quality ("low", "medium", "high")
                - texture_resolution: Texture resolution in pixels
                - output_format: Output format (glb, obj, ply)
                - seed: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - output_mesh_path: Path to generated mesh file
                - texture_resolution: Applied texture resolution
                - quality: Applied quality setting
                - generation_info: Additional generation metadata
        """
        # Validate inputs
        if "image_paths" not in inputs:
            raise ValueError("image_paths is required for image-to-mesh generation")

        image_paths = inputs["image_paths"]
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if not image_paths:
            raise ValueError("At least one image path must be provided")

        if len(image_paths) > self.max_images:
            raise ValueError(f"Too many images provided. Maximum: {self.max_images}")

        # Validate image files
        valid_image_paths = []
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")

            img_format = img_path.suffix.lower().lstrip(".")
            if img_format not in self.supported_input_formats:
                raise ValueError(f"Unsupported image format: {img_format}")

            valid_image_paths.append(str(img_path))

        output_format = self._validate_common_inputs(inputs)

        logger.info(
            f"Processing image-to-mesh request with {len(valid_image_paths)} images"
        )

        # This will be implemented by specific adapters
        response = self._create_common_response(inputs, output_format)
        response.update(
            {
                "input_images": valid_image_paths,
                "num_input_images": len(valid_image_paths),
                "seed": inputs.get("seed"),
                "generation_info": {
                    "input_type": "image",
                    "num_images": len(valid_image_paths),
                    "success": True,
                },
            }
        )

        return response

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats."""
        return {
            "input": self.supported_input_formats,
            "output": self.supported_output_formats,
        }
