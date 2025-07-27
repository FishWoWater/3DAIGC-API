"""
Mesh segmentation API endpoints.

Provides endpoints for partitioning 3D meshes into semantic parts.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.dependencies import get_scheduler
from api.routers.file_upload import resolve_file_id
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mesh-segmentation", tags=["mesh_segmentation"])


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


# Request models
class MeshSegmentationRequest(BaseModel):
    """Request for mesh segmentation"""

    mesh_path: Optional[str] = Field(None, description="Path to mesh file")
    mesh_base64: Optional[str] = Field(None, description="Base64 encoded mesh data")
    mesh_file_id: Optional[str] = Field(
        None, description="File ID from upload endpoint"
    )
    num_parts: int = Field(8, description="Target number of parts", ge=2, le=32)
    output_format: str = Field("glb", description="Output format")
    model_preference: str = Field(
        "partfield_mesh_segmentation", description="Model name for mesh segmentation"
    )

    @field_validator("mesh_file_id")
    @classmethod
    def validate_inputs(cls, v, info):
        mesh_path = info.data.get("mesh_path")
        mesh_base64 = info.data.get("mesh_base64")

        inputs_provided = sum(bool(x) for x in [mesh_path, mesh_base64, v])

        if inputs_provided == 0:
            raise ValueError(
                "One of mesh_path, mesh_base64, or mesh_file_id must be provided"
            )
        if inputs_provided > 1:
            raise ValueError(
                "Only one of mesh_path, mesh_base64, or mesh_file_id should be provided"
            )
        return v

    model_config = ConfigDict(protected_namespaces=("settings_",))


# Response models
class MeshSegmentationResponse(BaseModel):
    """Response for mesh segmentation requests"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


@router.post("/segment-mesh", response_model=MeshSegmentationResponse)
async def segment_mesh(
    request: MeshSegmentationRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Segment a 3D mesh into semantic parts.

    Args:
        request: Mesh segmentation parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the segmentation task
    """
    try:
        # Validate model preference
        validate_model_preference(
            request.model_preference, "mesh_segmentation", scheduler
        )

        # Validate input - either mesh_path, mesh_base64, or mesh_file_id must be provided
        if not any([request.mesh_path, request.mesh_base64, request.mesh_file_id]):
            raise HTTPException(
                status_code=400,
                detail="One of mesh_path, mesh_base64, or mesh_file_id must be provided",
            )

        # Process mesh input
        mesh_file_path = None

        if request.mesh_file_id:
            # Handle file ID
            mesh_file_path = resolve_file_id(request.mesh_file_id)
            if not mesh_file_path:
                raise HTTPException(
                    status_code=404, detail="Mesh file not found or expired"
                )
        elif request.mesh_base64:
            # Handle base64 encoded mesh
            import base64
            import tempfile

            try:
                mesh_data = base64.b64decode(request.mesh_base64)
                # Create temporary file
                temp_dir = Path(tempfile.gettempdir()) / "mesh_uploads"
                temp_dir.mkdir(exist_ok=True)

                # Try to determine file extension from data
                # This is a simple heuristic - in production, you'd want more robust detection
                if mesh_data.startswith(b"glTF"):
                    ext = ".glb"
                elif b"<mesh>" in mesh_data[:1000]:
                    ext = ".xml"
                else:
                    ext = ".obj"  # default

                temp_file = temp_dir / f"uploaded_mesh_{os.urandom(8).hex()}{ext}"
                temp_file.write_bytes(mesh_data)
                mesh_file_path = str(temp_file)

            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid mesh data: {str(e)}"
                )
        else:
            mesh_file_path = request.mesh_path

        # Validate mesh file exists
        if mesh_file_path and not Path(mesh_file_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Mesh file not found: {mesh_file_path}"
            )

        # Create job request
        job_request = JobRequest(
            feature="mesh_segmentation",
            inputs={
                "mesh_path": mesh_file_path,
                "num_parts": request.num_parts,
                "output_format": request.output_format,
            },
            model_preference=request.model_preference,
            priority=1,
            metadata={"feature_type": "mesh_segmentation"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return MeshSegmentationResponse(
            job_id=job_id,
            status="queued",
            message="Mesh segmentation job queued successfully",
        )

    except HTTPException:
        # Re-raise HTTP exceptions (including validation errors)
        raise
    except Exception as e:
        logger.error(f"Error scheduling mesh segmentation job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get supported input and output formats for mesh segmentation.

    Returns:
        Dictionary of supported formats
    """
    return {"input_formats": ["glb"], "output_formats": ["glb"]}
