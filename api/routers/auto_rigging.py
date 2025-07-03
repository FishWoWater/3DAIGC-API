"""
Auto-rigging API endpoints.

Provides endpoints for automatically adding bone structures to 3D meshes.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from api.dependencies import get_scheduler
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auto-rig", tags=["auto_rig"])


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
class AutoRigRequest(BaseModel):
    """Request for auto-rigging"""

    mesh_path: str = Field(..., description="Path to the input mesh file")
    rig_mode: str = Field("skeleton", description="Rig mode for auto-rigging")
    output_format: str = Field("fbx", description="Output format for rigged mesh")
    model_preference: str = Field(
        "unirig_auto_rig", description="Name of the auto-rigging model to use"
    )

    model_config = ConfigDict(protected_namespaces=("settings_",))


# Response models
class AutoRigResponse(BaseModel):
    """Response for auto-rigging requests"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


@router.post("/generate-rig", response_model=AutoRigResponse)
async def generate_rig(
    request: AutoRigRequest,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Generate bone structure for a 3D mesh.

    Args:
        request: Auto-rigging parameters
        scheduler: Model scheduler dependency

    Returns:
        Job information for the auto-rigging task
    """
    if request.rig_mode.lower() not in ["skeleton", "skin", "full"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid rig mode. Allowed: skeleton, skin, full",
        )

    try:
        # Validate model preference
        validate_model_preference(request.model_preference, "auto_rig", scheduler)

        # Validate rig type
        job_request = JobRequest(
            feature="auto_rig",
            inputs={
                "rig_mode": request.rig_mode.lower(),
                "mesh_path": request.mesh_path,
                "output_format": request.output_format,
            },
            model_preference=request.model_preference,
            priority=1,
            metadata={"feature_type": "auto_rig"},
        )

        job_id = await scheduler.schedule_job(job_request)

        return AutoRigResponse(
            job_id=job_id,
            status="queued",
            message="Auto-rigging job queued successfully",
        )
    except HTTPException as e:
        raise e

    except Exception as e:
        logger.error(f"Error scheduling auto-rig job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule job: {str(e)}")


@router.post("/upload-mesh")
async def upload_mesh_for_rigging(
    file: UploadFile = File(..., description="Mesh file (OBJ, GLB, FBX)"),
):
    """
    Upload a mesh file for auto-rigging.

    Args:
        file: Uploaded mesh file

    Returns:
        File path information
    """
    try:
        # Validate file format
        allowed_formats = {".obj", ".glb", ".fbx"}
        if file.filename is None:
            raise HTTPException(status_code=400, detail="File name is required")
        file_extension = "." + file.filename.split(".")[-1].lower()

        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {allowed_formats}",
            )

        # In a real implementation, save the file to storage
        # For now, return a mock path
        file_path = f"/uploads/meshes/{file.filename}"

        return {
            "filename": file.filename,
            "file_path": file_path,
            "size": file.size,
            "content_type": file.content_type,
        }

    except Exception as e:
        logger.error(f"Error uploading mesh file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get supported input and output formats for auto-rigging.

    Returns:
        Dictionary of supported formats
    """
    return {"input_formats": ["obj", "glb", "fbx"], "output_formats": ["fbx", "glb"]}
