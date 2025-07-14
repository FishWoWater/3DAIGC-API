"""System management and health check endpoints"""

import logging
import mimetypes
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from api.dependencies import get_current_settings, get_scheduler, verify_api_key
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler
from core.utils.file_utils import encode_file_to_base64, get_file_size_mb

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", summary="Health check")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time(),
    }


@router.get("/info", summary="System information")
async def system_info(
    settings=Depends(get_current_settings), _: bool = Depends(verify_api_key)
):
    """Get system information"""
    return {
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        },
        "application": {
            "version": "1.0.0",
            "environment": settings.environment,
            "debug": settings.debug,
        },
        "configuration": {
            "server": {
                "host": settings.server.host,
                "port": settings.server.port,
                "workers": settings.server.workers,
            },
            "gpu": {
                "auto_detect": settings.gpu.auto_detect,
                "memory_buffer": settings.gpu.memory_buffer,
            },
            "storage": {
                "input_dir": settings.storage.input_dir,
                "output_dir": settings.storage.output_dir,
            },
        },
    }


@router.get("/status", summary="Detailed system status")
async def system_status(
    settings=Depends(get_current_settings), _: bool = Depends(verify_api_key)
):
    """Get detailed system status including GPU information"""

    # Basic system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_usage": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "used": disk.used,
                "percent": (disk.used / disk.total) * 100,
            },
        },
        "gpu": [],
        "models": {"loaded": 0, "available": 0, "total_vram_used": 0},
        "queue": {"pending_jobs": 0, "processing_jobs": 0, "completed_jobs": 0},
    }

    # Try to get GPU information
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            status["gpu"].append(
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "memory_utilization": gpu.memoryUtil,
                    "gpu_utilization": gpu.load,
                    "temperature": gpu.temperature,
                }
            )
    except ImportError:
        status["gpu"] = "GPU monitoring not available (GPUtil not installed)"
    except Exception as e:
        status["gpu"] = f"Error getting GPU info: {str(e)}"

    return status


@router.get("/models", summary="List available models")
async def list_models(
    feature: Optional[str] = None,
    settings=Depends(get_current_settings),
    _: bool = Depends(verify_api_key),
):
    """List available models, optionally filtered by feature"""

    available_models = settings.list_available_models()

    if feature:
        if feature in available_models:
            return {"feature": feature, "models": available_models[feature]}
        else:
            raise HTTPException(
                status_code=404, detail=f"Feature '{feature}' not found"
            )

    return {
        "available_models": available_models,
        "total_features": len(available_models),
        "total_models": sum(len(models) for models in available_models.values()),
    }


@router.get("/features", summary="List supported features")
async def list_features(settings=Depends(get_current_settings)):
    """List all supported features"""
    available_models = settings.list_available_models()

    features = []
    for feature_name, models in available_models.items():
        features.append(
            {"name": feature_name, "model_count": len(models), "models": models}
        )

    return {"features": features, "total_features": len(features)}


@router.post("/shutdown", summary="Shutdown server")
async def shutdown_server(_: bool = Depends(verify_api_key)):
    """Shutdown the server (admin only)"""
    # This should only be available in development or with proper authentication
    import os
    import signal

    # Graceful shutdown
    os.kill(os.getpid(), signal.SIGTERM)

    return {"message": "Server shutdown initiated"}


@router.get("/logs", summary="Get recent logs")
async def get_logs(
    lines: int = Query(100, description="Number of recent lines to return"),
    level: Optional[str] = Query(
        None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ),
    logger_name: Optional[str] = Query(None, description="Filter by logger name"),
    since: Optional[str] = Query(
        None, description="Filter logs since timestamp (ISO format)"
    ),
    _: bool = Depends(verify_api_key),
):
    """Get recent log entries from log files"""

    try:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return {
                "error": "Logs directory not found",
                "message": "Logging may not be properly configured",
                "logs": [],
            }

        # Find log files
        log_files = list(logs_dir.glob("*.log"))
        if not log_files:
            return {"message": "No log files found", "logs": [], "available_files": []}

        # Get the most recent log file (app.log by default)
        main_log_file = logs_dir / "app.log"
        if not main_log_file.exists() and log_files:
            main_log_file = log_files[0]

        if not main_log_file.exists():
            return {
                "error": "Main log file not found",
                "available_files": [f.name for f in log_files],
                "logs": [],
            }

        # Read log file
        log_entries = []
        try:
            with open(main_log_file, "r", encoding="utf-8") as f:
                # Read last 'lines' number of lines efficiently
                file_lines = f.readlines()
                recent_lines = (
                    file_lines[-lines:] if len(file_lines) > lines else file_lines
                )

                for line in recent_lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse log entry
                    log_entry = _parse_log_line(line)

                    # Apply filters
                    if level and log_entry.get("level") != level.upper():
                        continue

                    if logger_name and logger_name not in log_entry.get("logger", ""):
                        continue

                    if since:
                        try:
                            from datetime import datetime

                            entry_time = datetime.fromisoformat(
                                log_entry.get("timestamp", "")
                            )
                            since_time = datetime.fromisoformat(since)
                            if entry_time < since_time:
                                continue
                        except (ValueError, TypeError):
                            pass  # Skip time filtering if parsing fails

                    log_entries.append(log_entry)

        except Exception as e:
            return {
                "error": f"Failed to read log file: {str(e)}",
                "logs": [],
                "file": str(main_log_file),
            }

        return {
            "logs": log_entries,
            "total_entries": len(log_entries),
            "file": str(main_log_file),
            "available_files": [f.name for f in log_files],
            "filters_applied": {
                "lines": lines,
                "level": level,
                "logger_name": logger_name,
                "since": since,
            },
        }

    except Exception as e:
        logger.error(f"Error retrieving logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")


def _parse_log_line(line: str) -> dict:
    """Parse a log line into structured data"""
    try:
        # Default log format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        # Example: "2024-01-01 12:00:00,000 - mylogger - INFO - This is a message"

        parts = line.split(" - ", 3)
        if len(parts) >= 4:
            timestamp_str = parts[0]
            logger_name = parts[1]
            level = parts[2]
            message = parts[3]

            return {
                "timestamp": timestamp_str,
                "logger": logger_name,
                "level": level,
                "message": message,
                "raw": line,
            }
        else:
            # Fallback for non-standard format
            return {
                "timestamp": "",
                "logger": "unknown",
                "level": "INFO",
                "message": line,
                "raw": line,
            }
    except Exception:
        # If parsing fails, return raw line
        return {
            "timestamp": "",
            "logger": "unknown",
            "level": "INFO",
            "message": line,
            "raw": line,
        }


@router.get("/logs/files", summary="List available log files")
async def list_log_files(_: bool = Depends(verify_api_key)):
    """List all available log files"""
    try:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return {"files": [], "message": "Logs directory not found"}

        log_files = []
        for log_file in logs_dir.glob("*.log"):
            try:
                stat = log_file.stat()
                log_files.append(
                    {
                        "name": log_file.name,
                        "path": str(log_file),
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    }
                )
            except Exception as e:
                log_files.append(
                    {
                        "name": log_file.name,
                        "path": str(log_file),
                        "error": f"Could not read file stats: {str(e)}",
                    }
                )

        return {
            "files": sorted(
                log_files, key=lambda x: x.get("modified", ""), reverse=True
            ),
            "total_files": len(log_files),
        }

    except Exception as e:
        logger.error(f"Error listing log files: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing log files: {str(e)}"
        )


@router.get("/logs/files/{filename}", summary="Get specific log file")
async def get_log_file(
    filename: str,
    lines: int = Query(100, description="Number of recent lines to return"),
    _: bool = Depends(verify_api_key),
):
    """Get contents of a specific log file"""
    try:
        logs_dir = Path("logs")
        log_file = logs_dir / filename

        # Security check - ensure the file is within logs directory
        if not str(log_file.resolve()).startswith(str(logs_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not log_file.exists():
            raise HTTPException(
                status_code=404, detail=f"Log file '{filename}' not found"
            )

        if not log_file.suffix == ".log":
            raise HTTPException(status_code=400, detail="Only .log files are allowed")

        log_entries = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                file_lines = f.readlines()
                recent_lines = (
                    file_lines[-lines:] if len(file_lines) > lines else file_lines
                )

                for line in recent_lines:
                    line = line.strip()
                    if line:
                        log_entries.append(_parse_log_line(line))

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to read log file: {str(e)}"
            )

        # Get file stats
        stat = log_file.stat()

        return {
            "filename": filename,
            "logs": log_entries,
            "total_entries": len(log_entries),
            "file_info": {
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "lines_requested": lines,
                "total_lines_in_file": len(file_lines)
                if "file_lines" in locals()
                else 0,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading log file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")


@router.delete("/logs/files/{filename}", summary="Delete log file")
async def delete_log_file(filename: str, _: bool = Depends(verify_api_key)):
    """Delete a specific log file (admin only)"""
    try:
        logs_dir = Path("logs")
        log_file = logs_dir / filename

        # Security check
        if not str(log_file.resolve()).startswith(str(logs_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not log_file.exists():
            raise HTTPException(
                status_code=404, detail=f"Log file '{filename}' not found"
            )

        if not log_file.suffix == ".log":
            raise HTTPException(
                status_code=400, detail="Only .log files can be deleted"
            )

        # Don't allow deletion of the main app.log while the server is running
        if filename == "app.log":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete main application log file while server is running",
            )

        log_file.unlink()

        return {
            "message": f"Log file '{filename}' deleted successfully",
            "deleted_file": filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting log file {filename}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting log file: {str(e)}"
        )


@router.get("/scheduler-status", summary="Get scheduler status")
async def get_scheduler_status(request: Request):
    """Get detailed scheduler and model status"""
    try:
        scheduler = await get_scheduler(request)
        status = await scheduler.get_system_status()

        return {
            "scheduler": {
                "running": True,
                "queue_status": status.get("queue", {}),
                "gpu_status": status.get("gpu", []),
                "models": status.get("models", {}),
                "features": status.get("features", {}),
            },
            "adapters_registered": len(status.get("models", {})),
            "active_jobs": status.get("queue", {}).get("processing_jobs", 0),
            "queued_jobs": status.get("queue", {}).get("queued_jobs", 0),
            "completed_jobs": status.get("queue", {}).get("completed_jobs", 0),
        }
    except Exception as e:
        return {"scheduler": {"running": False, "error": str(e)}}


@router.get("/jobs/history")
async def get_jobs_history(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    feature: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    scheduler: MultiprocessModelScheduler = Depends(get_scheduler),
):
    """
    Get historical jobs with filtering support.

    Args:
        limit: Maximum number of jobs to return (max 500)
        offset: Number of jobs to skip for pagination
        status: Filter by job status (queued, processing, completed, failed, cancelled)
        feature: Filter by feature type (e.g., text_to_textured_mesh)
        start_date: Filter jobs after this date (ISO format: 2024-01-01T00:00:00Z)
        end_date: Filter jobs before this date (ISO format: 2024-01-01T23:59:59Z)
        scheduler: Model scheduler dependency

    Returns:
        Paginated list of historical jobs
    """
    try:
        from datetime import datetime

        from core.scheduler.job_queue import JobStatus

        # Validate limit
        if limit > 500:
            limit = 500
        if limit < 1:
            limit = 1

        # Validate offset
        if offset < 0:
            offset = 0

        # Get all jobs from the database
        all_jobs = []

        # Get jobs from different statuses
        job_statuses_to_fetch = []
        if status:
            # If status filter is provided, only fetch that status
            try:
                status_enum = JobStatus(status.lower())
                job_statuses_to_fetch = [status_enum]
            except ValueError:
                # Invalid status provided
                return {
                    "jobs": [],
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "total": 0,
                        "has_more": False,
                    },
                    "filters": {
                        "status": status,
                        "feature": feature,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                    "timestamp": time.time(),
                    "error": f"Invalid status: {status}. Valid statuses: queued, processing, completed, failed, cancelled",
                }
        else:
            # Fetch all statuses
            job_statuses_to_fetch = list(JobStatus)

        # Collect all jobs
        for job_status in job_statuses_to_fetch:
            jobs = await scheduler.job_queue.get_jobs_by_status(job_status)
            for job in jobs:
                job_dict = job.to_dict()

                # Apply feature filter if provided
                if feature and job_dict.get("feature") != feature:
                    continue

                # Apply date filters if provided
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(
                            start_date.replace("Z", "+00:00")
                        )
                        job_created = datetime.fromisoformat(job_dict["created_at"])
                        if job_created < start_dt:
                            continue
                    except ValueError:
                        pass  # Skip invalid date format

                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                        job_created = datetime.fromisoformat(job_dict["created_at"])
                        if job_created > end_dt:
                            continue
                    except ValueError:
                        pass  # Skip invalid date format

                all_jobs.append(job_dict)

        # Sort by created_at (newest first)
        all_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Apply pagination
        total_jobs = len(all_jobs)
        start_idx = offset
        end_idx = min(offset + limit, total_jobs)
        paginated_jobs = all_jobs[start_idx:end_idx]

        return {
            "jobs": paginated_jobs,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_jobs,
                "has_more": end_idx < total_jobs,
            },
            "filters": {
                "status": status,
                "feature": feature,
                "start_date": start_date,
                "end_date": end_date,
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to get jobs history: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve jobs history: {str(e)}"
        )


@router.get("/jobs/{job_id}", summary="Get job status")
async def get_job_status(job_id: str, request: Request):
    """Get status of a specific job with visitable URLs for files"""
    try:
        scheduler = await get_scheduler(request)
        job_status = await scheduler.get_job_status(job_id)

        if job_status is None:
            raise HTTPException(status_code=404, detail="Job not found")

        # If job is completed and has results, convert file paths to URLs
        if job_status.get("status") == "completed" and job_status.get("result"):
            result = job_status["result"]

            # Convert mesh file path to URL
            mesh_path = None
            possible_mesh_keys = [
                "output_mesh_path",
                "mesh_path",
                "output_path",
                "file_path",
            ]

            for key in possible_mesh_keys:
                if key in result and result[key]:
                    mesh_path = result[key]
                    break

            if mesh_path and os.path.exists(mesh_path):
                # Create URL for mesh file
                mesh_url = f"{request.base_url}api/v1/system/jobs/{job_id}/download"
                result["mesh_url"] = mesh_url

                # Add file info
                file_stats = os.stat(mesh_path)
                result["mesh_file_info"] = {
                    "filename": os.path.basename(mesh_path),
                    "file_size_bytes": file_stats.st_size,
                    "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "content_type": get_content_type_for_file(mesh_path),
                    "file_extension": Path(mesh_path).suffix,
                }

            # Convert thumbnail path to URL
            thumbnail_path = result.get("thumbnail_path")
            if thumbnail_path and os.path.exists(thumbnail_path):
                # Create URL for thumbnail
                thumbnail_url = (
                    f"{request.base_url}api/v1/system/jobs/{job_id}/thumbnail"
                )
                result["thumbnail_url"] = thumbnail_url

                # Add thumbnail file info
                thumb_stats = os.stat(thumbnail_path)
                result["thumbnail_file_info"] = {
                    "filename": os.path.basename(thumbnail_path),
                    "file_size_bytes": thumb_stats.st_size,
                    "file_size_mb": round(thumb_stats.st_size / (1024 * 1024), 2),
                    "content_type": get_content_type_for_file(thumbnail_path),
                    "file_extension": Path(thumbnail_path).suffix,
                }

        return job_status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving job status: {str(e)}"
        )


@router.get("/adapters", summary="List registered adapters")
async def list_adapters(request: Request):
    """List all registered model adapters"""
    try:
        scheduler = await get_scheduler(request)
        status = await scheduler.get_system_status()

        adapters = []
        for model_id, model_info in status.get("models", {}).items():
            adapters.append(
                {
                    "model_id": model_id,
                    "feature_type": model_info.get("feature_type"),
                    "status": model_info.get("status"),
                    "vram_requirement": model_info.get("vram_requirement"),
                    "processing_count": model_info.get("processing_count", 0),
                    "supported_formats": model_info.get("supported_formats", {}),
                }
            )

        return {
            "adapters": adapters,
            "total_count": len(adapters),
            "by_feature": status.get("features", {}),
            "by_status": {
                "loaded": len([a for a in adapters if a["status"] == "loaded"]),
                "unloaded": len([a for a in adapters if a["status"] == "unloaded"]),
                "loading": len([a for a in adapters if a["status"] == "loading"]),
                "error": len([a for a in adapters if a["status"] == "error"]),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing adapters: {str(e)}")


def get_content_type_for_file(file_path: str) -> str:
    """Determine appropriate content type for a file"""
    file_ext = Path(file_path).suffix.lower()

    # Custom mappings for 3D formats
    content_type_mapping = {
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "application/wavefront-obj",
        ".fbx": "model/fbx",
        ".ply": "model/ply",
        ".stl": "model/stl",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".json": "application/json",
        ".txt": "text/plain",
        ".log": "text/plain",
    }

    if file_ext in content_type_mapping:
        return content_type_mapping[file_ext]

    # Fallback to mimetypes guess
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


@router.get("/jobs/{job_id}/download", summary="Download job result")
async def download_job_result(
    job_id: str,
    request: Request,
    format: Optional[str] = Query(
        None, description="Response format: 'file' (default) or 'base64'"
    ),
    filename: Optional[str] = Query(None, description="Custom filename for download"),
):
    """
    Download the result file of a completed job.

    Args:
        job_id: The job ID to download results for
        format: Response format ('file' for direct download, 'base64' for JSON response)
        filename: Optional custom filename for the download

    Returns:
        FileResponse for direct download or JSON with base64 data
    """
    try:
        scheduler = await get_scheduler(request)
        job_status = await scheduler.get_job_status(job_id)

        if job_status is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job_status.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not completed yet. Current status: {job_status.get('status')}",
            )

        result = job_status.get("result", {})
        if not result:
            raise HTTPException(
                status_code=404, detail="No result available for this job"
            )

        # Find the output file path - try multiple possible keys
        output_path = None
        possible_keys = ["output_mesh_path", "mesh_path", "output_path", "file_path"]

        for key in possible_keys:
            if key in result and result[key]:
                output_path = result[key]
                break

        if not output_path:
            raise HTTPException(
                status_code=404, detail="No output file path found in job result"
            )

        if not os.path.exists(output_path):
            raise HTTPException(
                status_code=404, detail=f"Output file not found at path: {output_path}"
            )

        # Determine the response format
        response_format = format or "file"

        if response_format == "base64":
            # Return base64 encoded data
            try:
                # Note: encode_file_to_base64 may have type issues, but implementation works
                base64_data = encode_file_to_base64(output_path)  # type: ignore
                file_size_mb = get_file_size_mb(output_path)

                return JSONResponse(
                    {
                        "job_id": job_id,
                        "filename": filename or os.path.basename(output_path),
                        "content_type": get_content_type_for_file(output_path),
                        "file_size_mb": file_size_mb,
                        "base64_data": base64_data,
                        "generation_info": result.get("generation_info", {}),
                        "download_time": datetime.utcnow().isoformat(),
                    }
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to encode file as base64: {str(e)}"
                )

        else:
            # Return file download
            download_filename = (
                filename or f"result_{job_id}_{os.path.basename(output_path)}"
            )
            content_type = get_content_type_for_file(output_path)

            return FileResponse(
                path=output_path,
                filename=download_filename,
                media_type=content_type,
                headers={
                    "X-Job-ID": job_id,
                    "X-Generation-Time": result.get("generation_info", {}).get(
                        "generation_time", "unknown"
                    ),
                    "X-File-Size": str(os.path.getsize(output_path)),
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@router.get("/jobs/{job_id}/thumbnail", summary="Download job thumbnail")
async def download_job_thumbnail(
    job_id: str,
    request: Request,
    format: Optional[str] = Query(
        None, description="Response format: 'file' (default) or 'base64'"
    ),
    filename: Optional[str] = Query(None, description="Custom filename for download"),
):
    """
    Download the thumbnail image of a completed job.

    Args:
        job_id: The job ID to download thumbnail for
        format: Response format ('file' for direct download, 'base64' for JSON response)
        filename: Optional custom filename for the download

    Returns:
        FileResponse for direct download or JSON with base64 data
    """
    try:
        scheduler = await get_scheduler(request)
        job_status = await scheduler.get_job_status(job_id)

        if job_status is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job_status.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not completed yet. Current status: {job_status.get('status')}",
            )

        result = job_status.get("result", {})
        if not result:
            raise HTTPException(
                status_code=404, detail="No result available for this job"
            )

        # Get thumbnail path
        thumbnail_path = result.get("thumbnail_path")
        if not thumbnail_path:
            raise HTTPException(
                status_code=404, detail="No thumbnail available for this job"
            )

        if not os.path.exists(thumbnail_path):
            raise HTTPException(
                status_code=404,
                detail=f"Thumbnail file not found at path: {thumbnail_path}",
            )

        # Determine the response format
        response_format = format or "file"

        if response_format == "base64":
            # Return base64 encoded data
            try:
                base64_data = encode_file_to_base64(thumbnail_path)  # type: ignore
                file_size_mb = get_file_size_mb(thumbnail_path)

                return JSONResponse(
                    {
                        "job_id": job_id,
                        "filename": filename or os.path.basename(thumbnail_path),
                        "content_type": get_content_type_for_file(thumbnail_path),
                        "file_size_mb": file_size_mb,
                        "base64_data": base64_data,
                        "generation_info": result.get("generation_info", {}),
                        "download_time": datetime.utcnow().isoformat(),
                    }
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to encode thumbnail as base64: {str(e)}",
                )

        else:
            # Return file download
            download_filename = (
                filename or f"thumbnail_{job_id}_{os.path.basename(thumbnail_path)}"
            )
            content_type = get_content_type_for_file(thumbnail_path)

            return FileResponse(
                path=thumbnail_path,
                filename=download_filename,
                media_type=content_type,
                headers={
                    "X-Job-ID": job_id,
                    "X-Thumbnail-Generated": str(
                        result.get("generation_info", {}).get(
                            "thumbnail_generated", "unknown"
                        )
                    ),
                    "X-File-Size": str(os.path.getsize(thumbnail_path)),
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error downloading thumbnail: {str(e)}"
        )


@router.get("/jobs/{job_id}/info", summary="Get job result information")
async def get_job_result_info(job_id: str, request: Request):
    """
    Get detailed information about a job's result without downloading the file.

    Args:
        job_id: The job ID to get information for

    Returns:
        Detailed job result information including file metadata
    """
    try:
        scheduler = await get_scheduler(request)
        job_status = await scheduler.get_job_status(job_id)

        if job_status is None:
            raise HTTPException(status_code=404, detail="Job not found")

        result = job_status.get("result", {})
        if not result:
            return {
                "job_id": job_id,
                "status": job_status.get("status"),
                "has_result": False,
                "message": "No result available",
            }

        # Get file information
        output_path = None
        possible_keys = ["output_mesh_path", "mesh_path", "output_path", "file_path"]

        for key in possible_keys:
            if key in result and result[key]:
                output_path = result[key]
                break

        file_info = {}
        if output_path and os.path.exists(output_path):
            file_stats = os.stat(output_path)
            file_info = {
                "filename": os.path.basename(output_path),
                "file_size_bytes": file_stats.st_size,
                "file_size_mb": file_stats.st_size / (1024 * 1024),
                "content_type": get_content_type_for_file(output_path),
                "file_extension": Path(output_path).suffix,
                "created_time": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(
                    file_stats.st_mtime
                ).isoformat(),
                "file_exists": True,
            }
        else:
            file_info = {"file_exists": False, "error": "Output file not found"}

        return {
            "job_id": job_id,
            "status": job_status.get("status"),
            "has_result": bool(result),
            "created_at": job_status.get("created_at"),
            "completed_at": job_status.get("completed_at"),
            "processing_time": job_status.get("processing_time"),
            "file_info": file_info,
            "generation_info": result.get("generation_info", {}),
            "result_metadata": {
                k: v
                for k, v in result.items()
                if k
                not in ["output_mesh_path", "mesh_path", "output_path", "file_path"]
            },
            "mesh_download_urls": {
                "direct_download": f"/api/v1/system/jobs/{job_id}/download",
                "base64_download": f"/api/v1/system/jobs/{job_id}/download?format=base64",
            },
            "thumbnail_download_urls": {
                "direct_download": f"/api/v1/system/jobs/{job_id}/thumbnail",
                "base64_download": f"/api/v1/system/jobs/{job_id}/thumbnail?format=base64",
            }
            if file_info.get("file_exists")
            else {},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting job result info: {str(e)}"
        )


@router.delete("/jobs/{job_id}/result", summary="Delete job result file")
async def delete_job_result(
    job_id: str, request: Request, _: bool = Depends(verify_api_key)
):
    """
    Delete the result file of a completed job to free up storage space.

    Args:
        job_id: The job ID to delete results for

    Returns:
        Confirmation of deletion
    """
    try:
        scheduler = await get_scheduler(request)
        job_status = await scheduler.get_job_status(job_id)

        if job_status is None:
            raise HTTPException(status_code=404, detail="Job not found")

        result = job_status.get("result", {})
        if not result:
            return {
                "job_id": job_id,
                "message": "No result file to delete",
                "deleted": False,
            }

        # Find and delete the output file
        output_path = None
        possible_keys = ["output_mesh_path", "mesh_path", "output_path", "file_path"]

        for key in possible_keys:
            if key in result and result[key]:
                output_path = result[key]
                break

        if not output_path:
            return {
                "job_id": job_id,
                "message": "No output file path found",
                "deleted": False,
            }

        if os.path.exists(output_path):
            file_size_mb = get_file_size_mb(output_path)
            os.remove(output_path)

            return {
                "job_id": job_id,
                "message": "Result file deleted successfully",
                "deleted": True,
                "freed_space_mb": file_size_mb,
                "deleted_file": os.path.basename(output_path),
            }
        else:
            return {
                "job_id": job_id,
                "message": "Result file does not exist",
                "deleted": False,
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting job result: {str(e)}"
        )


@router.get("/supported-formats", summary="Get supported formats")
async def get_supported_formats():
    """Get list of supported input and output formats"""
    return {
        "input_formats": {
            "text": ["string"],
            "image": ["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            "mesh": ["obj", "glb", "gltf", "ply", "stl", "fbx"],
            "base64": ["image/png", "image/jpeg", "model/gltf-binary"],
        },
        "output_formats": {
            "mesh": ["obj", "glb", "ply", "fbx"],
            "texture": ["png", "jpg"],
            "download": ["file", "base64"],
        },
        "content_types": {
            "mesh": {
                "glb": "model/gltf-binary",
                "gltf": "model/gltf+json",
                "obj": "application/wavefront-obj",
                "fbx": "model/fbx",
                "ply": "model/ply",
                "stl": "model/stl",
            },
            "image": {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
                "bmp": "image/bmp",
                "tiff": "image/tiff",
            },
        },
    }


@router.get("/available-adapters", summary="List available adapters from registry")
async def list_available_adapters(request: Request):
    """
    List all available adapters from the adapter registry.
    This provides more accurate information than the /models endpoint
    which only uses the models.yaml configuration.
    """
    try:
        scheduler = await get_scheduler(request)

        # Get adapter information from scheduler
        status = await scheduler.get_system_status()

        # Group adapters by feature type
        features = {}
        for model_id, model_info in status.get("models", {}).items():
            feature_type = model_info.get("feature_type")
            if feature_type:
                if feature_type not in features:
                    features[feature_type] = []
                features[feature_type].append(
                    {
                        "model_id": model_id,
                        "status": model_info.get("status"),
                        "vram_requirement": model_info.get("vram_requirement"),
                        "supported_formats": model_info.get("supported_formats", {}),
                    }
                )

        return {
            "features": features,
            "total_adapters": len(status.get("models", {})),
            "total_features": len(features),
        }
    except Exception as e:
        logger.error(f"Error retrieving adapter information: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving adapter information: {str(e)}"
        )


@router.get("/logs/config", summary="Get logging configuration")
async def get_logging_config(_: bool = Depends(verify_api_key)):
    """Get current logging configuration and levels"""
    try:
        # Get all loggers and their levels
        loggers_info = {}

        # Root logger
        root_logger = logging.getLogger()
        loggers_info["root"] = {
            "level": logging.getLevelName(root_logger.level),
            "handlers": [type(h).__name__ for h in root_logger.handlers],
            "effective_level": logging.getLevelName(root_logger.getEffectiveLevel()),
        }

        # Get all named loggers
        for name in logging.Logger.manager.loggerDict:
            logger_obj = logging.getLogger(name)
            if logger_obj.handlers or logger_obj.level != logging.NOTSET:
                loggers_info[name] = {
                    "level": logging.getLevelName(logger_obj.level),
                    "handlers": [type(h).__name__ for h in logger_obj.handlers],
                    "effective_level": logging.getLevelName(
                        logger_obj.getEffectiveLevel()
                    ),
                    "propagate": logger_obj.propagate,
                }

        return {
            "loggers": loggers_info,
            "available_levels": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "logs_directory": "logs",
            "config_source": "YAML configuration"
            if Path("config/logging.yaml").exists()
            else "Simple configuration",
        }

    except Exception as e:
        logger.error(f"Error retrieving logging config: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving logging config: {str(e)}"
        )


@router.post("/logs/test", summary="Generate test log entries")
async def generate_test_logs(
    level: str = Query("INFO", description="Log level for test entries"),
    count: int = Query(5, description="Number of test entries to generate"),
    _: bool = Depends(verify_api_key),
):
    """Generate test log entries for testing the logging system"""

    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level.upper() not in valid_levels:
        raise HTTPException(
            status_code=400, detail=f"Invalid log level. Must be one of: {valid_levels}"
        )

    if count < 1 or count > 50:
        raise HTTPException(status_code=400, detail="Count must be between 1 and 50")

    try:
        test_logger = logging.getLogger("api.test")
        log_level = getattr(logging, level.upper())

        test_messages = [
            "Test log entry for system verification",
            "Simulating application workflow",
            "Testing log aggregation and viewing",
            "Verifying log formatting and parsing",
            "Checking log file rotation and storage",
            "Testing different logger configurations",
            "Simulating error conditions for testing",
            "Validating log filtering and search",
            "Testing concurrent logging operations",
            "Verifying logging performance and reliability",
        ]

        generated_entries = []
        for i in range(count):
            message = f"{test_messages[i % len(test_messages)]} (#{i + 1})"
            test_logger.log(log_level, message)
            generated_entries.append(
                {
                    "level": level.upper(),
                    "message": message,
                    "logger": "api.test",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "message": f"Generated {count} test log entries at {level.upper()} level",
            "entries": generated_entries,
            "level": level.upper(),
            "count": count,
        }

    except Exception as e:
        logger.error(f"Error generating test logs: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating test logs: {str(e)}"
        )


@router.put("/logs/level", summary="Update logging level")
async def update_logging_level(
    logger_name: str = Query(
        ..., description="Logger name (use 'root' for root logger)"
    ),
    level: str = Query(..., description="New log level"),
    _: bool = Depends(verify_api_key),
):
    """Update logging level for a specific logger"""

    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level.upper() not in valid_levels:
        raise HTTPException(
            status_code=400, detail=f"Invalid log level. Must be one of: {valid_levels}"
        )

    try:
        if logger_name.lower() == "root":
            target_logger = logging.getLogger()
        else:
            target_logger = logging.getLogger(logger_name)

        old_level = logging.getLevelName(target_logger.level)
        new_level = getattr(logging, level.upper())
        target_logger.setLevel(new_level)

        return {
            "message": f"Updated logging level for '{logger_name}' from {old_level} to {level.upper()}",
            "logger_name": logger_name,
            "old_level": old_level,
            "new_level": level.upper(),
        }

    except Exception as e:
        logger.error(f"Error updating logging level: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error updating logging level: {str(e)}"
        )
