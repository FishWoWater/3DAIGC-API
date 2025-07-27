"""
Auto-rigging API endpoints.

Provides endpoints for automatically adding bone structures to 3D meshes.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.dependencies import get_scheduler
from api.routers.file_upload import resolve_file_id
from core.scheduler.job_queue import JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auto-rigging", tags=["auto_rigging"])


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

    mesh_path: Optional[str] = Field(None, description="Path to the input mesh file")
    mesh_file_id: Optional[str] = Field(
        None, description="File ID from upload endpoint"
    )
    rig_mode: str = Field("skeleton", description="Rig mode for auto-rigging")
    output_format: str = Field("fbx", description="Output format for rigged mesh")
    model_preference: str = Field(
        "unirig_auto_rig", description="Name of the auto-rigging model to use"
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v):
        allowed_formats = ["glb"]
        if v not in allowed_formats:
            raise ValueError(f"Output format must be one of: {allowed_formats}")
        return v

    @field_validator("mesh_file_id")
    @classmethod
    def validate_inputs(cls, v, info):
        mesh_path = info.data.get("mesh_path")

        inputs_provided = sum(bool(x) for x in [mesh_path, v])

        if inputs_provided == 0:
            raise ValueError("One of mesh_path or mesh_file_id must be provided")
        if inputs_provided > 1:
            raise ValueError("Only one of mesh_path or mesh_file_id should be provided")
        return v

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

        # Process mesh input
        mesh_file_path = None

        if request.mesh_file_id:
            # Handle file ID
            mesh_file_path = resolve_file_id(request.mesh_file_id)
            if not mesh_file_path:
                raise HTTPException(
                    status_code=404, detail="Mesh file not found or expired"
                )
        else:
            mesh_file_path = request.mesh_path

        # Validate mesh file exists
        if not mesh_file_path:
            raise HTTPException(
                status_code=400, detail="Mesh path or file ID must be provided"
            )

        # Validate rig type
        job_request = JobRequest(
            feature="auto_rig",
            inputs={
                "rig_mode": request.rig_mode.lower(),
                "mesh_path": mesh_file_path,
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


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get supported input and output formats for auto-rigging.

    Returns:
        Dictionary of supported formats
    """
    return {"input_formats": ["obj", "glb", "fbx"], "output_formats": ["glb"]}
