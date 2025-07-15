"""File handling utilities"""

import base64
import imghdr
import logging
import mimetypes
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import aiofiles
from fastapi import UploadFile
from PIL import Image

from .exceptions import FileUploadError

logger = logging.getLogger(__name__)

# Supported file formats
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tga"]
SUPPORTED_MESH_FORMATS = [".glb", ".obj", ".fbx", ".ply", ".stl", ".gltf"]
SUPPORTED_TEXTURE_FORMATS = [".jpg", ".jpeg", ".png", ".tga", ".exr", ".hdr"]

# Validation limits
MAX_IMAGE_RESOLUTION = (2048, 2048)  # Maximum image resolution
MAX_MESH_VERTICES = 30000  # Maximum number of vertices
MAX_MESH_FACES = 30000  # Maximum number of faces

# MIME type mappings
MIME_TYPE_MAPPING = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "model/gltf-binary": ".glb",
    "model/gltf+json": ".gltf",
    "application/wavefront-obj": ".obj",
    "model/fbx": ".fbx",
    "model/ply": ".ply",
    "model/stl": ".stl",
    "application/octet-stream": None,  # Generic binary
}


def generate_filename(original_filename: str, prefix: str = "") -> str:
    """Generate a unique filename with UUID"""
    file_ext = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}{file_ext}"
    return f"{unique_id}{file_ext}"


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)


def detect_file_type_from_content(file_path: str) -> str:
    """Detect file type from content analysis"""
    try:
        # Try to detect if it's an image
        img_type = imghdr.what(file_path)
        if img_type:
            return f"image/{img_type}"

        # Check MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type

        # Check file extension as fallback
        ext = Path(file_path).suffix.lower()
        if ext in SUPPORTED_IMAGE_FORMATS:
            return "image/generic"
        elif ext in SUPPORTED_MESH_FORMATS:
            return "model/generic"

        return "application/octet-stream"
    except Exception:
        return "application/octet-stream"


def validate_image_file(
    file_path: str, max_resolution: Tuple[int, int] = MAX_IMAGE_RESOLUTION
) -> Dict:
    """Validate and get info about an image file"""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            format_name = img.format
            mode = img.mode

            if width > max_resolution[0] or height > max_resolution[1]:
                raise ValueError(
                    f"Image resolution {width}x{height} exceeds maximum {max_resolution[0]}x{max_resolution[1]}"
                )

            return {
                "valid": True,
                "width": width,
                "height": height,
                "format": format_name,
                "mode": mode,
                "file_size_mb": get_file_size_mb(file_path),
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def validate_mesh_file(
    file_path: str,
    max_vertices: int = MAX_MESH_VERTICES,
    max_faces: int = MAX_MESH_FACES,
) -> Dict:
    """Validate and get info about a mesh file"""
    try:
        import trimesh

        # Load mesh using trimesh
        mesh = trimesh.load(file_path)

        # Handle scene objects
        if isinstance(mesh, trimesh.Scene):
            # Get the combined mesh from scene
            mesh = mesh.dump(concatenate=True)

        if not isinstance(mesh, trimesh.Trimesh):
            return {
                "valid": False,
                "error": f"Loaded object is not a valid mesh: {type(mesh)}",
            }

        # Get mesh statistics
        vertex_count = len(mesh.vertices)
        face_count = len(mesh.faces)

        # Check vertex count
        if vertex_count > max_vertices:
            raise ValueError(
                f"Mesh has {vertex_count} vertices, which exceeds maximum {max_vertices}"
            )

        # Check face count
        if face_count > max_faces:
            raise ValueError(
                f"Mesh has {face_count} faces, which exceeds maximum {max_faces}"
            )

        # Basic mesh validation
        if vertex_count == 0 or face_count == 0:
            raise ValueError("Mesh is empty (no vertices or faces)")

        return {
            "valid": True,
            "vertex_count": vertex_count,
            "face_count": face_count,
            "file_size_mb": get_file_size_mb(file_path),
            "is_watertight": bool(mesh.is_watertight),
        }

    except Exception as e:
        return {"valid": False, "error": str(e)}


def validate_base64_data(
    base64_data: str,
) -> Tuple[bool, Optional[str], Optional[bytes]]:
    """Validate base64 data and extract content type"""
    try:
        # Check if it has data URL prefix
        content_type = None
        if base64_data.startswith("data:"):
            if "," not in base64_data:
                return False, None, None
            header, base64_data = base64_data.split(",", 1)

            # Extract content type from header
            if ";" in header:
                content_type = header.split(";")[0].replace("data:", "")

        # Decode base64
        try:
            decoded_data = base64.b64decode(base64_data, validate=True)
        except Exception:
            return False, None, None

        # Validate it's not empty
        if len(decoded_data) == 0:
            return False, None, None

        return True, content_type, decoded_data
    except Exception:
        return False, None, None


async def save_upload_file(
    upload_file: UploadFile,
    destination_dir: str,
    max_size_mb: int = 100,
    validate_content: bool = True,
) -> Dict[str, Union[str, int, float, bool, Dict]]:
    """Save uploaded file to destination directory with enhanced validation"""
    try:
        # Create destination directory if it doesn't exist
        Path(destination_dir).mkdir(parents=True, exist_ok=True)

        # Validate file extension
        if upload_file.filename and not validate_file_extension(
            upload_file.filename, SUPPORTED_IMAGE_FORMATS + SUPPORTED_MESH_FORMATS
        ):
            raise FileUploadError(
                upload_file.filename,
                f"Unsupported file format. Supported: {SUPPORTED_IMAGE_FORMATS + SUPPORTED_MESH_FORMATS}",
            )

        # Generate unique filename
        filename = generate_filename(upload_file.filename or "upload.bin", "upload")
        file_path = Path(destination_dir) / filename

        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await upload_file.read(1024 * 1024):  # Read in 1MB chunks
                await f.write(chunk)

        # Check file size
        file_size_mb = get_file_size_mb(str(file_path))
        if file_size_mb > max_size_mb:
            os.remove(file_path)
            raise FileUploadError(
                upload_file.filename or "unknown",
                f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)",
            )

        # Validate content if requested
        validation_info = {}
        if validate_content and upload_file.filename:
            file_type = get_file_type_from_extension(upload_file.filename)
            if file_type == "image":
                validation_info = validate_image_file(str(file_path))
                if not validation_info.get("valid", False):
                    os.remove(file_path)
                    raise FileUploadError(
                        upload_file.filename or "unknown",
                        validation_info.get("error", "Invalid image file"),
                    )
            elif file_type == "mesh":
                validation_info = validate_mesh_file(str(file_path))
                if not validation_info.get("valid", False):
                    os.remove(file_path)
                    raise FileUploadError(
                        upload_file.filename or "unknown",
                        validation_info.get("error", "Invalid mesh file"),
                    )

        logger.info(f"Saved uploaded file: {file_path} ({file_size_mb:.1f}MB)")

        return {
            "file_path": str(file_path),
            "original_filename": upload_file.filename or "unknown",
            "saved_filename": filename,
            "file_size_mb": file_size_mb,
            "content_type": upload_file.content_type or "application/octet-stream",
            "file_type": get_file_type_from_extension(upload_file.filename or filename),
            "validation_info": validation_info,
        }

    except Exception as e:
        if isinstance(e, FileUploadError):
            raise
        raise FileUploadError(upload_file.filename or "unknown", str(e))


async def save_base64_file(
    base64_data: str, filename: str, destination_dir: str, validate_content: bool = True
) -> Dict[str, Union[str, int, float, bool, Dict]]:
    """Save base64 encoded data to file with enhanced validation"""
    try:
        # Create destination directory if it doesn't exist
        Path(destination_dir).mkdir(parents=True, exist_ok=True)

        # Validate base64 data
        is_valid, content_type, decoded_data = validate_base64_data(base64_data)
        if not is_valid:
            raise FileUploadError(filename, "Invalid base64 data")

        # Determine file extension from content type or filename
        file_ext = None
        if content_type and content_type in MIME_TYPE_MAPPING:
            file_ext = MIME_TYPE_MAPPING[content_type]

        if not file_ext:
            file_ext = Path(filename).suffix
            if not file_ext:
                # Try to detect from decoded data
                temp_path = Path(tempfile.mktemp())
                with open(temp_path, "wb") as f:
                    f.write(decoded_data or b"")
                detected_type = detect_file_type_from_content(str(temp_path))
                temp_path.unlink()
                if detected_type.startswith("image/"):
                    file_ext = ".png"  # Default to PNG for images
                else:
                    file_ext = ".bin"  # Binary for others

        # Create filename with proper extension
        base_name = Path(filename).stem if filename else "base64_file"
        final_filename = f"{base_name}{file_ext}"

        # Validate file extension
        if not validate_file_extension(
            final_filename, SUPPORTED_IMAGE_FORMATS + SUPPORTED_MESH_FORMATS
        ):
            raise FileUploadError(
                filename,
                f"Unsupported file format. Supported: {SUPPORTED_IMAGE_FORMATS + SUPPORTED_MESH_FORMATS}",
            )

        # Generate unique filename
        unique_filename = generate_filename(final_filename, "b64")
        file_path = Path(destination_dir) / unique_filename

        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(decoded_data or b"")

        file_size_mb = get_file_size_mb(str(file_path))

        # Validate content if requested
        validation_info = {}
        if validate_content:
            file_type = get_file_type_from_extension(final_filename)
            if file_type == "image":
                validation_info = validate_image_file(str(file_path))
                if not validation_info.get("valid", False):
                    os.remove(file_path)
                    raise FileUploadError(
                        filename, validation_info.get("error", "Invalid image file")
                    )
            elif file_type == "mesh":
                validation_info = validate_mesh_file(str(file_path))
                if not validation_info.get("valid", False):
                    os.remove(file_path)
                    raise FileUploadError(
                        filename, validation_info.get("error", "Invalid mesh file")
                    )

        logger.info(f"Saved base64 file: {file_path} ({file_size_mb:.1f}MB)")

        return {
            "file_path": str(file_path),
            "original_filename": filename,
            "saved_filename": unique_filename,
            "file_size_mb": file_size_mb,
            "content_type": content_type or "application/octet-stream",
            "file_type": get_file_type_from_extension(final_filename),
            "validation_info": validation_info,
        }

    except Exception as e:
        if isinstance(e, FileUploadError):
            raise
        raise FileUploadError(filename, f"Failed to save base64 data: {str(e)}")


async def process_mixed_input(
    file_upload: Optional[UploadFile] = None,
    base64_data: Optional[str] = None,
    file_path: Optional[str] = None,
    destination_dir: str = "",
    max_size_mb: int = 100,
) -> Dict[str, Union[str, int, float, bool, Dict]]:
    """Process input that can be upload, base64, or existing file path"""
    if sum(bool(x) for x in [file_upload, base64_data, file_path]) != 1:
        raise ValueError(
            "Exactly one of file_upload, base64_data, or file_path must be provided"
        )

    if file_upload:
        return await save_upload_file(file_upload, destination_dir, max_size_mb)
    elif base64_data:
        filename = "uploaded_file"
        return await save_base64_file(base64_data, filename, destination_dir)
    elif file_path:
        # Validate existing file path
        if not os.path.exists(file_path):
            raise FileUploadError(file_path, "File does not exist")

        file_size_mb = get_file_size_mb(file_path)
        file_type = get_file_type_from_extension(file_path)

        validation_info = {}
        if file_type == "image":
            validation_info = validate_image_file(file_path)
            if not validation_info.get("valid", False):
                raise FileUploadError(
                    file_path, validation_info.get("error", "Invalid image file")
                )
        elif file_type == "mesh":
            validation_info = validate_mesh_file(file_path)
            if not validation_info.get("valid", False):
                raise FileUploadError(
                    file_path, validation_info.get("error", "Invalid mesh file")
                )

        return {
            "file_path": file_path,
            "original_filename": os.path.basename(file_path),
            "saved_filename": os.path.basename(file_path),
            "file_size_mb": file_size_mb,
            "content_type": detect_file_type_from_content(file_path),
            "file_type": file_type,
            "validation_info": validation_info,
        }

    # This should never be reached due to the validation at the start
    raise ValueError("Invalid input combination")


def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 string with data URL format"""
    try:
        content_type = detect_file_type_from_content(file_path)

        with open(file_path, "rb") as f:
            encoded_data = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{content_type};base64,{encoded_data}"
    except Exception as e:
        raise FileUploadError(file_path, f"Failed to encode file to base64: {str(e)}")


def cleanup_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")


async def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """Clean up files older than specified hours"""
    cleanup_count = 0
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

    try:
        for file_path in Path(directory).iterdir():
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleanup_count += 1
                        logger.debug(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to cleanup old file {file_path}: {str(e)}"
                        )

    except Exception as e:
        logger.error(f"Error during cleanup of directory {directory}: {str(e)}")

    if cleanup_count > 0:
        logger.info(f"Cleaned up {cleanup_count} old files from {directory}")

    return cleanup_count


def ensure_directory(directory: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_safe_filename(filename: str) -> str:
    """Get safe filename by removing/replacing problematic characters"""
    import re

    # Remove path separators and other problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Limit length
    if len(safe_name) > 255:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[: 255 - len(ext)] + ext

    return safe_name


def copy_file(src: str, dst: str) -> None:
    """Copy file from source to destination"""
    try:
        shutil.copy2(src, dst)
        logger.debug(f"Copied file from {src} to {dst}")
    except Exception as e:
        raise FileUploadError(src, f"Failed to copy file: {str(e)}")


def move_file(src: str, dst: str) -> None:
    """Move file from source to destination"""
    try:
        shutil.move(src, dst)
        logger.debug(f"Moved file from {src} to {dst}")
    except Exception as e:
        raise FileUploadError(src, f"Failed to move file: {str(e)}")


async def create_temp_directory() -> str:
    """Create a temporary directory"""
    temp_dir = tempfile.mkdtemp(prefix="p3d_")
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension against allowed list"""
    file_ext = Path(filename).suffix.lower()
    return file_ext in [ext.lower() for ext in allowed_extensions]


def get_file_type_from_extension(filename: str) -> str:
    """Get file type based on extension"""
    ext = Path(filename).suffix.lower()

    image_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]
    mesh_exts = [".glb", ".obj", ".fbx", ".ply", ".stl"]
    texture_exts = [".jpg", ".jpeg", ".png", ".tga", ".exr", ".hdr"]

    if ext in image_exts:
        return "image"
    elif ext in mesh_exts:
        return "mesh"
    elif ext in texture_exts:
        return "texture"
    else:
        return "unknown"
