"""
Mesh generation API endpoints.

Provides endpoints for generating 3D meshes from various inputs including text, images,
and combinations of both. Enhanced to support file uploads, base64 encoding, and proper
result downloading.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.dependencies import get_scheduler
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler
from core.utils.file_utils import save_base64_file, save_upload_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mesh-generation", tags=["mesh_generation"])


def validate_model_preference(
    model_preference: str, feature: str, scheduler: MultiprocessModelScheduler
) -> None:
    """
    Validate that the model preference is available for the given feature.

    Args:
        model_preference: The preferred model ID
        feature: The feature type for the job
        scheduler: The model scheduler instance

    Raises:
        HTTPException: If the model preference is invalid
    """
    if not scheduler.validate_model_preference(model_preference, feature):
        available_models = scheduler.get_available_models(feature)
        feature_models = available_models.get(feature, [])

        if not feature_models:
            raise HTTPException(
                status_code=400,
                detail=f"No models available for feature '{feature}'. Please check if models are registered.",
            )

        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_preference}' is not available for feature '{feature}'. "
            f"Available models: {feature_models}",
        )


# Enhanced Request models with file upload support
class TextToRawMeshRequest(BaseModel):
    """Request for text-to-mesh generation"""

    text_prompt: str = Field(..., description="Text description for mesh generation")
    output_format: str = Field("glb", description="Output mesh format")
    model_preference: str = Field(
        "trellis_text_to_raw_mesh", description="Model name for mesh generation"
    )

    model_config = ConfigDict(protected_namespaces=("settings_",))

    @field_validator("output_format")
    def validate_output_format(cls, v):
        allowed_formats = ["glb", "obj", "fbx", "ply"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v


class TextToTexturedMeshRequest(TextToRawMeshRequest):
    """Request for text-to-textured-mesh generation"""

    texture_prompt: str = Field(
        "", description="Text description for texture generation"
    )
    texture_resolution: int = Field(
        1024, description="Texture resolution", ge=256, le=4096
    )


class TextMeshPaintingRequest(BaseModel):
    """Request for text-based mesh painting"""

    text_prompt: str = Field(..., description="Text description for painting")
    mesh_path: Optional[str] = Field(
        None, description="Path to the input mesh file (for local files)"
    )
    mesh_base64: Optional[str] = Field(None, description="Base64 encoded mesh data")
    texture_resolution: int = Field(
        1024, description="Texture resolution", ge=256, le=4096
    )
    output_format: str = Field("glb", description="Output mesh format")
    model_preference: str = Field(
        "trellis_text_to_textured_mesh", description="Model name for mesh generation"
    )

    @field_validator("output_format")
    def validate_output_format(cls, v):
        allowed_formats = ["glb", "obj", "fbx", "ply"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v

    @field_validator("mesh_base64")
    def validate_inputs(cls, v, values):
        mesh_path = values.get("mesh_path")
        if not mesh_path and not v:
            raise ValueError("Either mesh_path or mesh_base64 must be provided")
        if mesh_path and v:
            raise ValueError("Only one of mesh_path or mesh_base64 should be provided")
        return v

    model_config = ConfigDict(protected_namespaces=("settings_",))


class ImageToRawMeshRequest(BaseModel):
    """Request for image-to-mesh generation"""

    image_path: Optional[str] = Field(
        None, description="Path to the input image (for local files)"
    )
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    output_format: str = Field("glb", description="Output mesh format")
    model_preference: str = Field(
        "trellis_image_to_raw_mesh", description="Model name for mesh generation"
    )

    @field_validator("output_format")
    def validate_output_format(cls, v):
        allowed_formats = ["glb", "obj", "fbx", "ply"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v

    @field_validator("image_base64")
    def validate_inputs(cls, v, values):
        image_path = values.get("image_path")
        if not image_path and not v:
            raise ValueError("Either image_path or image_base64 must be provided")
        if image_path and v:
            raise ValueError(
                "Only one of image_path or image_base64 should be provided"
            )
        return v

    model_config = ConfigDict(protected_namespaces=("settings_",))


class ImageToTexturedMeshRequest(BaseModel):
    """Request for image-to-textured-mesh generation"""

    image_path: Optional[str] = Field(None, description="Path to the input image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    texture_image_path: Optional[str] = Field(
        None, description="Path to the texture image"
    )
    texture_image_base64: Optional[str] = Field(
        None, description="Base64 encoded texture image"
    )
    texture_resolution: int = Field(
        1024, description="Texture resolution", ge=256, le=4096
    )
    output_format: str = Field("glb", description="Output mesh format")
    model_preference: str = Field(
        "trellis_image_to_textured_mesh", description="Model name for mesh generation"
    )

    @field_validator("output_format")
    def validate_output_format(cls, v):
        allowed_formats = ["glb", "obj", "fbx", "ply"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v

    @field_validator("image_base64")
    def validate_inputs(cls, v, values):
        image_path = values.get("image_path")
        if not image_path and not v:
            raise ValueError("Either image_path or image_base64 must be provided")
        if image_path and v:
            raise ValueError(
                "Only one of image_path or image_base64 should be provided"
            )
        return v

    model_config = ConfigDict(protected_namespaces=("settings_",))


class ImageMeshPaintingRequest(BaseModel):
    """Request for image-based mesh painting"""

    image_path: Optional[str] = Field(None, description="Path to the input image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    mesh_path: Optional[str] = Field(None, description="Path to the input mesh file")
    mesh_base64: Optional[str] = Field(None, description="Base64 encoded mesh data")
    texture_resolution: int = Field(
        1024, description="Texture resolution", ge=256, le=4096
    )
    output_format: str = Field("glb", description="Output mesh format")
    model_preference: str = Field(
        "trellis_image_mesh_painting", description="Model name for mesh generation"
    )

    @field_validator("output_format")
    def validate_output_format(cls, v):
        allowed_formats = ["glb", "obj", "fbx", "ply"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v

    @field_validator("mesh_base64")
    def validate_inputs(cls, v, values):
        image_path = values.get("image_path")
        image_base64 = values.get("image_base64")
        mesh_path = values.get("mesh_path")

        if not image_path and not image_base64:
            raise ValueError("Either image_path or image_base64 must be provided")
        if image_path and image_base64:
            raise ValueError(
                "Only one of image_path or image_base64 should be provided"
            )
        if not mesh_path and not v:
            raise ValueError("Either mesh_path or mesh_base64 must be provided")
        if mesh_path and v:
            raise ValueError("Only one of mesh_path or mesh_base64 should be provided")
        return v

    model_config = ConfigDict(protected_namespaces=("settings_",))


# Part Completion Request
class PartCompletionRequest(BaseModel):
    """Request for part completion"""

    mesh_path: Optional[str] = Field(None, description="Path to the input mesh file")
    mesh_base64: Optional[str] = Field(None, description="Base64 encoded mesh data")
    output_format: str = Field("glb", description="Output mesh format")
    model_preference: str = Field(
        "holopart_part_completion", description="Model name for part completion"
    )

    model_config = ConfigDict(protected_namespaces=("settings_",))


# Enhanced Response models
class MeshGenerationResponse(BaseModel):
    """Response for mesh generation requests"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


class FileUploadResponse(BaseModel):
    """Response for file upload operations"""

    file_id: str = Field(..., description="Unique file identifier")
    original_filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Detected file type")
    file_size_mb: float = Field(..., description="File size in MB")
    message: str = Field(..., description="Upload status message")


# Helper function to process file inputs
async def process_file_input(
    file_path: Optional[str] = None,
    base64_data: Optional[str] = None,
    upload_file: Optional[UploadFile] = None,
    input_type: str = "image",
) -> str:
    """Process various file input formats and return the processed file path"""

    if not any([file_path, base64_data, upload_file]):
        raise HTTPException(status_code=400, detail=f"No {input_type} input provided")

    if sum(bool(x) for x in [file_path, base64_data, upload_file]) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Multiple {input_type} inputs provided. Only one allowed.",
        )

    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp(prefix="mesh_gen_")

    try:
        if file_path:
            # Validate existing file path
            if not Path(file_path).exists():
                raise HTTPException(
                    status_code=404, detail=f"{input_type.title()} file not found"
                )
            return str(file_path)

        elif base64_data:
            # Process base64 data
            file_info = await save_base64_file(
                base64_data, f"input_{input_type}", temp_dir
            )
            return str(file_info["file_path"])

        elif upload_file:
            # Process uploaded file
            file_info = await save_upload_file(upload_file, temp_dir)
            return str(file_info["file_path"])
        else:
            raise HTTPException(status_code=400, detail="No input provided")

    except Exception as e:
        logger.error(f"Error processing {input_type} input: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Error processing {input_type}: {str(e)}"
        )


# Text-to-mesh endpoints
@router.post("/text-to-raw-mesh", response_model=MeshGenerationResponse)
async def text_to_raw_mesh(
    mesh_request: TextToRawMeshRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Generate a 3D mesh from text description.

    Args:
        mesh_request: Text-to-mesh generation parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the mesh generation task
    """
    try:
        # Validate model preference
        validate_model_preference(
            mesh_request.model_preference, "text_to_raw_mesh", scheduler
        )

        job_request = JobRequest(
            feature="text_to_raw_mesh",
            inputs={
                "text_prompt": mesh_request.text_prompt,
                "output_format": mesh_request.output_format,
            },
            model_preference=mesh_request.model_preference,
            priority=1,
            metadata={"feature_type": "text_to_raw_mesh"},
        )
        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Text-to-raw-mesh generation job queued successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including validation errors)
        raise
    except Exception as e:
        logger.error(f"Error scheduling text-to-raw-mesh job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.post("/text-to-textured-mesh", response_model=MeshGenerationResponse)
async def text_to_textured_mesh(
    mesh_request: TextToTexturedMeshRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Generate a 3D textured mesh from text description.

    Args:
        mesh_request: Text-to-textured-mesh generation parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the mesh generation task
    """
    try:
        # Validate model preference
        validate_model_preference(
            mesh_request.model_preference, "text_to_textured_mesh", scheduler
        )

        job_request = JobRequest(
            feature="text_to_textured_mesh",
            inputs={
                "text_prompt": mesh_request.text_prompt,
                "texture_prompt": mesh_request.texture_prompt,
                "output_format": mesh_request.output_format,
                "texture_resolution": mesh_request.texture_resolution,
            },
            model_preference=mesh_request.model_preference,
            priority=1,
            metadata={"feature_type": "text_to_textured_mesh"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Text-to-textured-mesh generation job queued successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including validation errors)
        raise
    except Exception as e:
        logger.error(f"Error scheduling text-to-textured-mesh job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


# Text-based mesh painting endpoint (supports both file path and base64)
@router.post("/text-mesh-painting", response_model=MeshGenerationResponse)
async def text_mesh_painting(
    mesh_request: TextMeshPaintingRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Texture a 3D mesh from text description.

    Args:
        mesh_request: Text-based mesh painting parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the mesh painting task
    """
    try:
        # Validate model preference
        validate_model_preference(
            mesh_request.model_preference, "text_mesh_painting", scheduler
        )

        # Process mesh input
        mesh_file_path = await process_file_input(
            file_path=mesh_request.mesh_path,
            base64_data=mesh_request.mesh_base64,
            input_type="mesh",
        )

        job_request = JobRequest(
            feature="text_mesh_painting",
            inputs={
                "text_prompt": mesh_request.text_prompt,
                "mesh_path": mesh_file_path,
                "output_format": mesh_request.output_format,
                "texture_resolution": mesh_request.texture_resolution,
            },
            model_preference=mesh_request.model_preference,
            priority=1,
            metadata={"feature_type": "text_mesh_painting"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Text-based mesh painting job queued successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including validation errors)
        raise
    except Exception as e:
        logger.error(f"Error scheduling text-mesh-painting job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


# Image-to-mesh endpoints (supports both file path and base64)
@router.post("/image-to-raw-mesh", response_model=MeshGenerationResponse)
async def image_to_raw_mesh(
    mesh_request: ImageToRawMeshRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Generate a 3D mesh from image.

    Args:
        mesh_request: Image-to-mesh generation parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the mesh generation task
    """
    try:
        # Validate model preference
        validate_model_preference(
            mesh_request.model_preference, "image_to_raw_mesh", scheduler
        )

        # Process image input
        image_file_path = await process_file_input(
            file_path=mesh_request.image_path,
            base64_data=mesh_request.image_base64,
            input_type="image",
        )

        job_request = JobRequest(
            feature="image_to_raw_mesh",
            inputs={
                "image_path": image_file_path,
                "output_format": mesh_request.output_format,
            },
            model_preference=mesh_request.model_preference,
            priority=1,
            metadata={"feature_type": "image_to_raw_mesh"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Image-to-raw-mesh generation job queued successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including validation errors)
        raise
    except Exception as e:
        logger.error(f"Error scheduling image-to-raw-mesh job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.post("/image-to-textured-mesh", response_model=MeshGenerationResponse)
async def image_to_textured_mesh(
    mesh_request: ImageToTexturedMeshRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Generate a 3D textured mesh from image.

    Args:
        mesh_request: Image-to-textured-mesh generation parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the mesh generation task
    """
    try:
        # Validate model preference
        validate_model_preference(
            mesh_request.model_preference, "image_to_textured_mesh", scheduler
        )

        # Process main image input
        image_file_path = await process_file_input(
            file_path=mesh_request.image_path,
            base64_data=mesh_request.image_base64,
            input_type="image",
        )

        # Process texture image if provided
        texture_image_path = None
        if mesh_request.texture_image_path or mesh_request.texture_image_base64:
            texture_image_path = await process_file_input(
                file_path=mesh_request.texture_image_path,
                base64_data=mesh_request.texture_image_base64,
                input_type="texture_image",
            )

        job_request = JobRequest(
            feature="image_to_textured_mesh",
            inputs={
                "image_path": image_file_path,
                "texture_image_path": texture_image_path,
                "output_format": mesh_request.output_format,
                "texture_resolution": mesh_request.texture_resolution,
            },
            model_preference=mesh_request.model_preference,
            priority=1,
            metadata={"feature_type": "image_to_textured_mesh"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Image-to-textured-mesh generation job queued successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including validation errors)
        raise
    except Exception as e:
        logger.error(f"Error scheduling image-to-textured-mesh job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.post("/image-mesh-painting", response_model=MeshGenerationResponse)
async def image_mesh_painting(
    mesh_request: ImageMeshPaintingRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Texture a 3D mesh using an image.

    Args:
        mesh_request: Image-based mesh painting parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the mesh painting task
    """
    try:
        # Validate model preference
        validate_model_preference(
            mesh_request.model_preference, "image_mesh_painting", scheduler
        )

        # Process image input
        image_file_path = await process_file_input(
            file_path=mesh_request.image_path,
            base64_data=mesh_request.image_base64,
            input_type="image",
        )

        # Process mesh input
        mesh_file_path = await process_file_input(
            file_path=mesh_request.mesh_path,
            base64_data=mesh_request.mesh_base64,
            input_type="mesh",
        )

        job_request = JobRequest(
            feature="image_mesh_painting",
            inputs={
                "image_path": image_file_path,
                "mesh_path": mesh_file_path,
                "output_format": mesh_request.output_format,
                "texture_resolution": mesh_request.texture_resolution,
            },
            model_preference=mesh_request.model_preference,
            priority=1,
            metadata={"feature_type": "image_mesh_painting"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Image-based mesh painting job queued successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including validation errors)
        raise
    except Exception as e:
        logger.error(f"Error scheduling image-mesh-painting job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.post("/part-completion", response_model=MeshGenerationResponse)
async def part_completion(
    mesh_request: PartCompletionRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Complete a part of a 3D mesh.
    """
    try:
        # Validate model preference
        validate_model_preference(
            mesh_request.model_preference, "part_completion", scheduler
        )

        # Process mesh input
        mesh_file_path = await process_file_input(
            file_path=mesh_request.mesh_path,
            base64_data=mesh_request.mesh_base64,
            input_type="mesh",
        )

        job_request = JobRequest(
            feature="part_completion",
            inputs={
                "mesh_path": mesh_file_path,
                "output_format": mesh_request.output_format,
            },
            model_preference=mesh_request.model_preference,
            priority=1,
            metadata={"feature_type": "part_completion"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Part completion job queued successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including validation errors)
        raise
    except Exception as e:
        logger.error(f"Error scheduling part completion job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.post("/upload-image", response_model=FileUploadResponse)
async def upload_image_for_mesh(
    file: UploadFile = File(..., description="Image file (PNG, JPG, JPEG, WebP)"),
):
    """
    Upload an image file for later use in mesh generation.

    Args:
        file: Image file to upload

    Returns:
        File upload information including file ID for later reference
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Create upload directory
        upload_dir = Path("uploads/images")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_info = await save_upload_file(file, str(upload_dir), max_size_mb=50)

        return FileUploadResponse(
            file_id=str(file_info["saved_filename"]),
            original_filename=str(file_info["original_filename"]),
            file_type=str(file_info["file_type"]),
            file_size_mb=float(file_info["file_size_mb"]),
            message="Image uploaded successfully",
        )

    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload-mesh", response_model=FileUploadResponse)
async def upload_mesh_for_painting(
    file: UploadFile = File(..., description="Mesh file (GLB, OBJ, PLY)"),
):
    """
    Upload a mesh file for later use in mesh painting.

    Args:
        file: Mesh file to upload

    Returns:
        File upload information including file ID for later reference
    """
    try:
        # Validate file type
        allowed_extensions = [".glb", ".obj", ".ply", ".fbx"]
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File must be a mesh file. Allowed: {allowed_extensions}",
            )

        # Create upload directory
        upload_dir = Path("uploads/meshes")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_info = await save_upload_file(file, str(upload_dir), max_size_mb=200)

        return FileUploadResponse(
            file_id=str(file_info["saved_filename"]),
            original_filename=str(file_info["original_filename"]),
            file_type=str(file_info["file_type"]),
            file_size_mb=float(file_info["file_size_mb"]),
            message="Mesh uploaded successfully",
        )

    except Exception as e:
        logger.error(f"Error uploading mesh: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# Enhanced file endpoints using upload IDs
@router.post("/text-mesh-painting-upload", response_model=MeshGenerationResponse)
async def text_mesh_painting_with_upload(
    text_prompt: str = Form(..., description="Text description for painting"),
    texture_resolution: int = Form(1024, description="Texture resolution"),
    output_format: str = Form("glb", description="Output format"),
    mesh_file: UploadFile = File(..., description="Mesh file to paint"),
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Texture a 3D mesh using uploaded file and text description.
    """
    try:
        # Process uploaded mesh
        mesh_file_path = await process_file_input(
            upload_file=mesh_file, input_type="mesh"
        )

        job_request = JobRequest(
            feature="text_mesh_painting",
            inputs={
                "text_prompt": text_prompt,
                "mesh_path": mesh_file_path,
                "output_format": output_format,
                "texture_resolution": texture_resolution,
            },
            priority=1,
            metadata={"feature_type": "text_mesh_painting"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Text-based mesh painting job with upload queued successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling text-mesh-painting with upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.post("/image-to-raw-mesh-upload", response_model=MeshGenerationResponse)
async def image_to_raw_mesh_with_upload(
    output_format: str = Form("glb", description="Output format"),
    image_file: UploadFile = File(..., description="Image file for mesh generation"),
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Generate a 3D mesh from uploaded image file.
    """
    try:
        # Process uploaded image
        image_file_path = await process_file_input(
            upload_file=image_file, input_type="image"
        )

        job_request = JobRequest(
            feature="image_to_raw_mesh",
            inputs={
                "image_path": image_file_path,
                "output_format": output_format,
            },
            priority=1,
            metadata={"feature_type": "image_to_raw_mesh"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Image-to-raw-mesh generation job with upload queued successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling image-to-raw-mesh with upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


# Utility endpoints
@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported input and output formats"""
    return {
        "input_formats": {
            "text": ["string"],
            "image": ["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            "mesh": ["obj", "glb", "gltf", "ply", "stl", "fbx"],
            "base64": ["image/png", "image/jpeg", "model/gltf-binary"],
        },
        "output_formats": {
            "mesh": ["obj", "glb", "ply", "fbx"],
            "texture": ["png", "jpg"],
        },
        "upload_limits": {
            "image_max_size_mb": 50,
            "mesh_max_size_mb": 200,
            "image_max_resolution": [4096, 4096],
        },
    }


@router.get("/generation-presets")
async def get_generation_presets():
    """Get available generation presets and their parameters"""
    return {
        "quality_presets": {
            "low": {
                "description": "Fast generation with basic quality",
                "texture_resolution": 512,
                "vertex_limit": 5000,
            },
            "medium": {
                "description": "Balanced generation quality and speed",
                "texture_resolution": 1024,
                "vertex_limit": 10000,
            },
            "high": {
                "description": "High quality generation (slower)",
                "texture_resolution": 2048,
                "vertex_limit": 20000,
            },
        },
        "style_presets": {
            "realistic": "Photorealistic style with detailed textures",
            "cartoon": "Stylized cartoon appearance",
            "lowpoly": "Low polygon count geometric style",
            "artistic": "Artistic interpretation with creative liberty",
        },
    }
