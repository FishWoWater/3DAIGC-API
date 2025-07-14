"""
File upload API endpoints.

Provides endpoints for uploading images and meshes with unique identifiers
that can be used in other API endpoints.
"""

import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core.utils.file_utils import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_MESH_FORMATS,
    FileUploadError,
    save_upload_file,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/file-upload", tags=["file_upload"])

# Configuration
UPLOAD_BASE_DIR = Path("uploads")
UPLOAD_BASE_DIR.mkdir(exist_ok=True)

# File metadata storage (in production, this would be a database)
file_metadata: Dict[str, Dict] = {}


class FileUploadResponse(BaseModel):
    """Response for file upload requests"""

    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Type of file (image/mesh)")
    file_size_mb: float = Field(..., description="File size in MB")
    upload_time: datetime = Field(..., description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="File expiration time")


class FileMetadataResponse(BaseModel):
    """Response for file metadata requests"""

    file_id: str = Field(..., description="Unique identifier for the file")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Type of file (image/mesh)")
    file_size_mb: float = Field(..., description="File size in MB")
    upload_time: datetime = Field(..., description="Upload timestamp")
    expires_at: Optional[datetime] = Field(None, description="File expiration time")
    is_available: bool = Field(..., description="Whether the file is still available")


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Validate file type based on extension"""
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_types


def generate_file_id() -> str:
    """Generate unique file identifier"""
    return str(uuid.uuid4())


def store_file_metadata(file_id: str, file_info: Dict) -> None:
    """Store file metadata (in production, this would be a database operation)"""
    file_metadata[file_id] = {
        "file_id": file_id,
        "filename": file_info["original_filename"],
        "file_path": file_info["file_path"],
        "file_type": file_info["file_type"],
        "file_size_mb": file_info["file_size_mb"],
        "upload_time": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=24),  # 24 hours default
        "is_available": True,
    }


def get_file_metadata(file_id: str) -> Optional[Dict]:
    """Get file metadata by ID"""
    return file_metadata.get(file_id)


def get_file_path(file_id: str) -> Optional[str]:
    """Get file path by ID"""
    metadata = get_file_metadata(file_id)
    if metadata and metadata["is_available"]:
        file_path = metadata["file_path"]
        if os.path.exists(file_path):
            return file_path
    return None


async def upload_file_with_validation(
    file: UploadFile,
    file_type: str,
    allowed_extensions: List[str],
    max_size_mb: int = 100,
) -> Dict:
    """Upload file with validation and return metadata"""

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not validate_file_type(file.filename, allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed extensions: {allowed_extensions}",
        )

    # Generate file ID and create directory
    file_id = generate_file_id()
    upload_dir = (
        UPLOAD_BASE_DIR / file_type / file_id[:2]
    )  # Use first 2 chars for subdirectory
    upload_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save file
        file_info = await save_upload_file(
            file, str(upload_dir), max_size_mb=max_size_mb, validate_content=True
        )

        # Store metadata
        store_file_metadata(file_id, file_info)

        logger.info(f"Uploaded {file_type} file {file_id}: {file.filename}")

        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_type": file_type,
            "file_size_mb": file_info["file_size_mb"],
            "upload_time": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24),
        }
    except FileUploadError as e:
        logger.error(f"Failed to upload {file_type} file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Failed to upload {file_type} file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/image", response_model=FileUploadResponse)
async def upload_image(
    file: UploadFile = File(..., description="Image file to upload"),
):
    """
    Upload an image file.

    Returns a unique file ID that can be used in other API endpoints.
    Supported formats: PNG, JPG, JPEG, WebP, BMP, TIFF
    """
    return await upload_file_with_validation(
        file, "image", SUPPORTED_IMAGE_FORMATS, max_size_mb=50
    )


@router.post("/mesh", response_model=FileUploadResponse)
async def upload_mesh(file: UploadFile = File(..., description="Mesh file to upload")):
    """
    Upload a mesh file.

    Returns a unique file ID that can be used in other API endpoints.
    Supported formats: GLB, OBJ, FBX, PLY, STL, GLTF
    """
    return await upload_file_with_validation(
        file, "mesh", SUPPORTED_MESH_FORMATS, max_size_mb=200
    )


@router.get("/metadata/{file_id}", response_model=FileMetadataResponse)
async def get_file_metadata_endpoint(file_id: str):
    """
    Get metadata for an uploaded file.

    Args:
        file_id: Unique file identifier

    Returns:
        File metadata including availability status
    """
    metadata = get_file_metadata(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    # Check if file still exists
    is_available = metadata["is_available"] and os.path.exists(metadata["file_path"])

    return FileMetadataResponse(
        file_id=metadata["file_id"],
        filename=metadata["filename"],
        file_type=metadata["file_type"],
        file_size_mb=metadata["file_size_mb"],
        upload_time=metadata["upload_time"],
        expires_at=metadata["expires_at"],
        is_available=is_available,
    )


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """
    Delete an uploaded file.

    Args:
        file_id: Unique file identifier

    Returns:
        Deletion confirmation
    """
    metadata = get_file_metadata(file_id)

    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Remove file from disk
        if os.path.exists(metadata["file_path"]):
            os.remove(metadata["file_path"])

        # Mark as unavailable
        metadata["is_available"] = False

        logger.info(f"Deleted file {file_id}: {metadata['filename']}")

        return {
            "file_id": file_id,
            "message": "File deleted successfully",
            "filename": metadata["filename"],
        }

    except Exception as e:
        logger.error(f"Failed to delete file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@router.get("/list")
async def list_uploaded_files(file_type: Optional[str] = None, limit: int = 100):
    """
    List uploaded files.

    Args:
        file_type: Optional filter by file type (image/mesh)
        limit: Maximum number of files to return

    Returns:
        List of uploaded files
    """
    files = []

    for metadata in file_metadata.values():
        if file_type and metadata["file_type"] != file_type:
            continue

        if len(files) >= limit:
            break

        files.append(
            {
                "file_id": metadata["file_id"],
                "filename": metadata["filename"],
                "file_type": metadata["file_type"],
                "file_size_mb": metadata["file_size_mb"],
                "upload_time": metadata["upload_time"],
                "is_available": metadata["is_available"]
                and os.path.exists(metadata["file_path"]),
            }
        )

    return {"files": files, "count": len(files), "total_files": len(file_metadata)}


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get supported file formats for upload.

    Returns:
        Dictionary of supported formats and limits
    """
    return {
        "image": {
            "formats": SUPPORTED_IMAGE_FORMATS,
            "max_size_mb": 50,
            "description": "Supported image formats for upload",
        },
        "mesh": {
            "formats": SUPPORTED_MESH_FORMATS,
            "max_size_mb": 200,
            "description": "Supported mesh formats for upload",
        },
        "retention": {
            "default_hours": 24,
            "description": "Files are automatically deleted after 24 hours",
        },
    }


# Utility function to be used by other modules
def resolve_file_id(file_id: str) -> Optional[str]:
    """
    Resolve a file ID to its actual file path.

    This function can be imported and used by other modules
    to convert file IDs to file paths.
    """
    return get_file_path(file_id)
