# 3D Generative Models Backend API Documentation

## Overview

This API provides scalable 3D AI model inference capabilities with VRAM-aware scheduling. The backend supports multiple 3D generation features including mesh generation, texturing, segmentation, and auto-rigging.

**Base URL**: `http://localhost:8000` (or your configured host/port)
**API Version**: v1
**Documentation**: `/docs` (Swagger UI) or `/redoc` (ReDoc)

## Authentication

Some endpoints require API key authentication. Include the API key in the request headers:
```
Authorization: Bearer YOUR_API_KEY
```

## Response Format

All API responses follow a consistent format:

### Success Response
```json
{
  "job_id": "unique_job_identifier",
  "status": "queued|processing|completed|failed",
  "message": "descriptive_message"
}
```

### Error Response
```json
{
  "error": "ERROR_CODE",
  "message": "Human readable error message",
  "detail": "Additional error details"
}
```

## Rate Limits

- File uploads: 50MB for images, 200MB for meshes
- Concurrent jobs: Managed by VRAM-aware scheduler
- Texture resolution: 256-4096 pixels

---

## System Management Endpoints

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Basic health check for the API
- **Authentication**: None required
- **Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "uptime": 1234567890
}
```

### System Information
- **URL**: `/api/v1/system/info`
- **Method**: `GET`
- **Description**: Get detailed system information
- **Authentication**: Required
- **Response**:
```json
{
  "system": {
    "platform": "Linux-5.4.0",
    "python_version": "3.10.0",
    "cpu_count": 8,
    "memory_total": 34359738368,
    "memory_available": 16777216000
  },
  "application": {
    "version": "1.0.0",
    "environment": "development",
    "debug": true
  },
  "configuration": {
    "server": {"host": "0.0.0.0", "port": 8000, "workers": 4},
    "gpu": {"auto_detect": true, "memory_buffer": 1024},
    "storage": {"input_dir": "inputs", "output_dir": "outputs"}
  }
}
```

### System Status
- **URL**: `/api/v1/system/status`
- **Method**: `GET`
- **Description**: Get detailed system status including GPU information
- **Authentication**: Required
- **Response**:
```json
{
  "timestamp": "2024-01-01T00:00:00.000Z",
  "system": {
    "cpu_usage": 45.2,
    "memory": {
      "total": 34359738368,
      "available": 16777216000,
      "used": 17582522368,
      "percent": 51.2
    },
    "disk": {
      "total": 1000000000000,
      "free": 500000000000,
      "used": 500000000000,
      "percent": 50.0
    }
  },
  "gpu": [
    {
      "id": 0,
      "name": "NVIDIA RTX 4090",
      "memory_total": 24576,
      "memory_used": 8192,
      "memory_free": 16384,
      "memory_utilization": 0.33,
      "gpu_utilization": 0.75,
      "temperature": 65
    }
  ],
  "models": {"loaded": 3, "available": 10, "total_vram_used": 8192},
  "queue": {"pending_jobs": 2, "processing_jobs": 1, "completed_jobs": 15}
}
```

### List Available Models
- **URL**: `/api/v1/system/models`
- **Method**: `GET`
- **Description**: List available models, optionally filtered by feature
- **Authentication**: Required
- **Query Parameters**:
  - `feature` (optional): Filter by specific feature type
- **Response**:
```json
{
  "available_models": {
    "text_to_raw_mesh": ["trellis_text_to_raw_mesh"],
    "image_to_raw_mesh": ["trellis_image_to_raw_mesh"],
    "mesh_segmentation": ["partfield_mesh_segmentation"],
    "auto_rig": ["unirig_auto_rig"]
  },
  "total_features": 4,
  "total_models": 4
}
```

### List Supported Features
- **URL**: `/api/v1/system/features`
- **Method**: `GET`
- **Description**: List all supported features
- **Authentication**: None required
- **Response**:
```json
{
  "features": [
    {
      "name": "text_to_raw_mesh",
      "model_count": 1,
      "models": ["trellis_text_to_raw_mesh"]
    },
    {
      "name": "mesh_segmentation",
      "model_count": 1,
      "models": ["partfield_mesh_segmentation"]
    }
  ],
  "total_features": 2
}
```

### Get Job Status
- **URL**: `/api/v1/system/jobs/{job_id}`
- **Method**: `GET`
- **Description**: Get status of a specific job
- **Authentication**: None required
- **Path Parameters**:
  - `job_id`: Unique job identifier
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "completed",
  "created_at": "2024-01-01T00:00:00.000Z",
  "completed_at": "2024-01-01T00:05:00.000Z",
  "processing_time": 300.5,
  "result": {
    "output_mesh_path": "/outputs/mesh_123456.glb",
    "generation_info": {
      "model_used": "trellis_text_to_raw_mesh",
      "parameters": {"text_prompt": "A red car"}
    }
  }
}
```

### Download Job Result
- **URL**: `/api/v1/system/jobs/{job_id}/download`
- **Method**: `GET`
- **Description**: Download the result file of a completed job
- **Authentication**: None required
- **Path Parameters**:
  - `job_id`: Unique job identifier
- **Query Parameters**:
  - `format` (optional): `file` (default) or `base64`
  - `filename` (optional): Custom filename for download
- **Response**: Binary file download or base64 encoded data

### Get Job Result Information
- **URL**: `/api/v1/system/jobs/{job_id}/info`
- **Method**: `GET`
- **Description**: Get detailed information about job result without downloading
- **Authentication**: None required
- **Path Parameters**:
  - `job_id`: Unique job identifier
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "completed",
  "file_info": {
    "file_exists": true,
    "file_size_mb": 2.5,
    "file_format": "glb",
    "content_type": "model/gltf-binary"
  },
  "generation_info": {
    "model_used": "trellis_text_to_raw_mesh",
    "processing_time": 300.5
  },
  "download_urls": {
    "direct_download": "/api/v1/system/jobs/job_123456/download",
    "base64_download": "/api/v1/system/jobs/job_123456/download?format=base64"
  }
}
```

### Delete Job Result
- **URL**: `/api/v1/system/jobs/{job_id}/result`
- **Method**: `DELETE`
- **Description**: Delete the result file of a completed job to free storage
- **Authentication**: Required
- **Path Parameters**:
  - `job_id`: Unique job identifier
- **Response**:
```json
{
  "job_id": "job_123456",
  "message": "Result file deleted successfully",
  "deleted": true,
  "freed_space_mb": 2.5,
  "deleted_file": "mesh_123456.glb"
}
```

### Scheduler Status
- **URL**: `/api/v1/system/scheduler-status`
- **Method**: `GET`
- **Description**: Get detailed scheduler and model status
- **Authentication**: None required
- **Response**:
```json
{
  "scheduler": {
    "running": true,
    "queue_status": {
      "queued_jobs": 2,
      "processing_jobs": 1,
      "completed_jobs": 15
    },
    "gpu_status": [
      {
        "id": 0,
        "memory_used": 8192,
        "memory_total": 24576
      }
    ],
    "models": {
      "trellis_text_to_raw_mesh": {"status": "loaded", "vram_usage": 4096}
    }
  },
  "adapters_registered": 4,
  "active_jobs": 1,
  "queued_jobs": 2,
  "completed_jobs": 15
}
```

### Get Supported Formats
- **URL**: `/api/v1/system/supported-formats`
- **Method**: `GET`
- **Description**: Get list of supported input and output formats
- **Authentication**: None required
- **Response**:
```json
{
  "input_formats": {
    "text": ["string"],
    "image": ["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    "mesh": ["obj", "glb", "gltf", "ply", "stl", "fbx"],
    "base64": ["image/png", "image/jpeg", "model/gltf-binary"]
  },
  "output_formats": {
    "mesh": ["obj", "glb", "ply", "fbx"],
    "texture": ["png", "jpg"],
    "download": ["file", "base64"]
  },
  "content_types": {
    "mesh": {
      "glb": "model/gltf-binary",
      "obj": "application/wavefront-obj",
      "fbx": "model/fbx"
    }
  }
}
```

### Get Generation Presets
- **URL**: `/api/v1/system/generation-presets`
- **Method**: `GET`
- **Description**: Get available generation presets and their parameters
- **Authentication**: None required
- **Response**:
```json
{
  "quality_presets": {
    "low": {
      "description": "Fast generation with basic quality",
      "texture_resolution": 512,
      "vertex_limit": 5000
    },
    "medium": {
      "description": "Balanced generation quality and speed",
      "texture_resolution": 1024,
      "vertex_limit": 10000
    },
    "high": {
      "description": "High quality generation (slower)",
      "texture_resolution": 2048,
      "vertex_limit": 20000
    }
  },
  "style_presets": {
    "realistic": "Photorealistic style with detailed textures",
    "cartoon": "Stylized cartoon appearance",
    "lowpoly": "Low polygon count geometric style",
    "artistic": "Artistic interpretation with creative liberty"
  }
}
```

### List Available Adapters
- **URL**: `/api/v1/system/available-adapters`
- **Method**: `GET`
- **Description**: List all available adapters from the adapter registry
- **Authentication**: None required
- **Response**:
```json
{
  "features": {
    "text_to_raw_mesh": [
      {
        "model_id": "trellis_text_to_raw_mesh",
        "status": "loaded",
        "vram_requirement": 4096,
        "supported_formats": {
          "input": ["text"],
          "output": ["glb", "obj", "fbx", "ply"]
        }
      }
    ]
  },
  "total_adapters": 4,
  "total_features": 4
}
```

---

## Mesh Generation Endpoints

### Text to Raw Mesh
- **URL**: `/api/v1/mesh-generation/text-to-raw-mesh`
- **Method**: `POST`
- **Description**: Generate a 3D mesh from text description
- **Authentication**: None required
- **Request Body**:
```json
{
  "text_prompt": "A red sports car",
  "output_format": "glb",
  "model_preference": "trellis_text_to_raw_mesh"
}
```
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Text-to-mesh generation job queued successfully"
}
```

### Text to Textured Mesh
- **URL**: `/api/v1/mesh-generation/text-to-textured-mesh`
- **Method**: `POST`
- **Description**: Generate a textured 3D mesh from text description
- **Authentication**: None required
- **Request Body**:
```json
{
  "text_prompt": "A red sports car",
  "texture_prompt": "shiny metallic paint",
  "texture_resolution": 1024,
  "output_format": "glb",
  "model_preference": "trellis_text_to_textured_mesh"
}
```
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Text-to-textured-mesh generation job queued successfully"
}
```

### Text Mesh Painting
- **URL**: `/api/v1/mesh-generation/text-mesh-painting`
- **Method**: `POST`
- **Description**: Apply texture to an existing mesh using text description
- **Authentication**: None required
- **Request Body**:
```json
{
  "text_prompt": "rusty metal texture",
  "mesh_path": "/path/to/mesh.glb",
  "mesh_base64": null,
  "texture_resolution": 1024,
  "output_format": "glb",
  "model_preference": "trellis_text_to_textured_mesh"
}
```
- **Note**: Provide either `mesh_path` or `mesh_base64`, not both
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Text-based mesh painting job queued successfully"
}
```

### Image to Raw Mesh
- **URL**: `/api/v1/mesh-generation/image-to-raw-mesh`
- **Method**: `POST`
- **Description**: Generate a 3D mesh from an image
- **Authentication**: None required
- **Request Body**:
```json
{
  "image_path": "/path/to/image.jpg",
  "image_base64": null,
  "output_format": "glb",
  "model_preference": "trellis_image_to_raw_mesh"
}
```
- **Note**: Provide either `image_path` or `image_base64`, not both
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Image-to-mesh generation job queued successfully"
}
```

### Image to Textured Mesh
- **URL**: `/api/v1/mesh-generation/image-to-textured-mesh`
- **Method**: `POST`
- **Description**: Generate a textured 3D mesh from an image
- **Authentication**: None required
- **Request Body**:
```json
{
  "image_path": "/path/to/image.jpg",
  "image_base64": null,
  "texture_image_path": "/path/to/texture.jpg",
  "texture_image_base64": null,
  "texture_resolution": 1024,
  "output_format": "glb",
  "model_preference": "trellis_image_to_textured_mesh"
}
```
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Image-to-textured-mesh generation job queued successfully"
}
```

### Image Mesh Painting
- **URL**: `/api/v1/mesh-generation/image-mesh-painting`
- **Method**: `POST`
- **Description**: Apply texture to an existing mesh using an image
- **Authentication**: None required
- **Request Body**:
```json
{
  "image_path": "/path/to/texture.jpg",
  "image_base64": null,
  "mesh_path": "/path/to/mesh.glb",
  "mesh_base64": null,
  "texture_resolution": 1024,
  "output_format": "glb",
  "model_preference": "trellis_image_mesh_painting"
}
```
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Image-based mesh painting job queued successfully"
}
```

### Part Completion
- **URL**: `/api/v1/mesh-generation/part-completion`
- **Method**: `POST`
- **Description**: Complete missing parts of a 3D mesh
- **Authentication**: None required
- **Request Body**:
```json
{
  "mesh_path": "/path/to/incomplete_mesh.glb",
  "mesh_base64": null,
  "output_format": "glb",
  "model_preference": "holopart_part_completion"
}
```
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Part completion job queued successfully"
}
```

### Upload Image
- **URL**: `/api/v1/mesh-generation/upload-image`
- **Method**: `POST`
- **Description**: Upload an image file for later use in mesh generation
- **Authentication**: None required
- **Request Body**: `multipart/form-data`
  - `file`: Image file (PNG, JPG, JPEG, WebP, max 50MB)
- **Response**:
```json
{
  "file_id": "img_123456",
  "original_filename": "my_image.jpg",
  "file_type": "image/jpeg",
  "file_size_mb": 2.5,
  "message": "Image uploaded successfully"
}
```

### Upload Mesh
- **URL**: `/api/v1/mesh-generation/upload-mesh`
- **Method**: `POST`
- **Description**: Upload a mesh file for later use in mesh painting
- **Authentication**: None required
- **Request Body**: `multipart/form-data`
  - `file`: Mesh file (GLB, OBJ, PLY, max 200MB)
- **Response**:
```json
{
  "file_id": "mesh_123456",
  "original_filename": "my_mesh.glb",
  "file_type": "model/gltf-binary",
  "file_size_mb": 15.2,
  "message": "Mesh uploaded successfully"
}
```

### Text Mesh Painting with Upload
- **URL**: `/api/v1/mesh-generation/text-mesh-painting-upload`
- **Method**: `POST`
- **Description**: Apply texture to an uploaded mesh using text description
- **Authentication**: None required
- **Request Body**: `multipart/form-data`
  - `text_prompt`: Text description for painting
  - `texture_resolution`: Texture resolution (default: 1024)
  - `output_format`: Output format (default: "glb")
  - `mesh_file`: Mesh file to paint
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Text-based mesh painting job with upload queued successfully"
}
```

### Image to Raw Mesh with Upload
- **URL**: `/api/v1/mesh-generation/image-to-raw-mesh-upload`
- **Method**: `POST`
- **Description**: Generate a 3D mesh from an uploaded image
- **Authentication**: None required
- **Request Body**: `multipart/form-data`
  - `output_format`: Output format (default: "glb")
  - `image_file`: Image file for mesh generation
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Image-to-raw-mesh generation job with upload queued successfully"
}
```

### Get Mesh Generation Supported Formats
- **URL**: `/api/v1/mesh-generation/supported-formats`
- **Method**: `GET`
- **Description**: Get supported formats for mesh generation
- **Authentication**: None required
- **Response**:
```json
{
  "input_formats": {
    "text": ["string"],
    "image": ["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
    "mesh": ["obj", "glb", "gltf", "ply", "stl", "fbx"],
    "base64": ["image/png", "image/jpeg", "model/gltf-binary"]
  },
  "output_formats": {
    "mesh": ["obj", "glb", "ply", "fbx"],
    "texture": ["png", "jpg"]
  },
  "upload_limits": {
    "image_max_size_mb": 50,
    "mesh_max_size_mb": 200,
    "image_max_resolution": [4096, 4096]
  }
}
```

### Get Mesh Generation Presets
- **URL**: `/api/v1/mesh-generation/generation-presets`
- **Method**: `GET`
- **Description**: Get available generation presets and their parameters
- **Authentication**: None required
- **Response**:
```json
{
  "quality_presets": {
    "low": {
      "description": "Fast generation with basic quality",
      "texture_resolution": 512,
      "vertex_limit": 5000
    },
    "medium": {
      "description": "Balanced generation quality and speed",
      "texture_resolution": 1024,
      "vertex_limit": 10000
    },
    "high": {
      "description": "High quality generation (slower)",
      "texture_resolution": 2048,
      "vertex_limit": 20000
    }
  },
  "style_presets": {
    "realistic": "Photorealistic style with detailed textures",
    "cartoon": "Stylized cartoon appearance",
    "lowpoly": "Low polygon count geometric style",
    "artistic": "Artistic interpretation with creative liberty"
  }
}
```

---

## Mesh Segmentation Endpoints

### Segment Mesh
- **URL**: `/api/v1/mesh-segmentation/segment-mesh`
- **Method**: `POST`
- **Description**: Segment a 3D mesh into semantic parts
- **Authentication**: None required
- **Request Body**:
```json
{
  "mesh_path": "/path/to/mesh.glb",
  "mesh_base64": null,
  "num_parts": 8,
  "output_format": "glb",
  "model_preference": "partfield_mesh_segmentation"
}
```
- **Parameters**:
  - `mesh_path` or `mesh_base64`: Input mesh (provide only one)
  - `num_parts`: Target number of parts (2-32)
  - `output_format`: Output format
  - `model_preference`: Model to use for segmentation
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Mesh segmentation job queued successfully"
}
```

### Upload Mesh for Segmentation
- **URL**: `/api/v1/mesh-segmentation/upload-mesh`
- **Method**: `POST`
- **Description**: Upload a mesh file for segmentation
- **Authentication**: None required
- **Request Body**: `multipart/form-data`
  - `file`: Mesh file (GLB format only)
- **Response**:
```json
{
  "filename": "mesh.glb",
  "file_path": "/uploads/meshes/mesh.glb",
  "size": 1024000,
  "content_type": "model/gltf-binary"
}
```

### Get Mesh Segmentation Supported Formats
- **URL**: `/api/v1/mesh-segmentation/supported-formats`
- **Method**: `GET`
- **Description**: Get supported formats for mesh segmentation
- **Authentication**: None required
- **Response**:
```json
{
  "input_formats": ["glb"],
  "output_formats": ["glb", "json"]
}
```

---

## Auto Rigging Endpoints

### Generate Rig
- **URL**: `/api/v1/auto-rigging/generate-rig`
- **Method**: `POST`
- **Description**: Generate bone structure for a 3D mesh
- **Authentication**: None required
- **Request Body**:
```json
{
  "mesh_path": "/path/to/mesh.glb",
  "rig_mode": "skeleton",
  "output_format": "fbx",
  "model_preference": "unirig_auto_rig"
}
```
- **Parameters**:
  - `mesh_path`: Path to input mesh file
  - `rig_mode`: Rig mode (`skeleton`, `skin`, or `full`)
  - `output_format`: Output format
  - `model_preference`: Model to use for rigging
- **Response**:
```json
{
  "job_id": "job_123456",
  "status": "queued",
  "message": "Auto-rigging job queued successfully"
}
```

### Upload Mesh for Rigging
- **URL**: `/api/v1/auto-rigging/upload-mesh`
- **Method**: `POST`
- **Description**: Upload a mesh file for auto-rigging
- **Authentication**: None required
- **Request Body**: `multipart/form-data`
  - `file`: Mesh file (OBJ, GLB, FBX)
- **Response**:
```json
{
  "filename": "character.fbx",
  "file_path": "/uploads/meshes/character.fbx",
  "size": 2048000,
  "content_type": "model/fbx"
}
```

### Get Auto Rigging Supported Formats
- **URL**: `/api/v1/auto-rigging/supported-formats`
- **Method**: `GET`
- **Description**: Get supported formats for auto-rigging
- **Authentication**: None required
- **Response**:
```json
{
  "input_formats": ["obj", "glb", "fbx"],
  "output_formats": ["fbx", "glb"]
}
```

---

## Workflow Examples

### Example 1: Text to Mesh Generation
```bash
# 1. Generate mesh from text
curl -X POST "http://localhost:8000/api/v1/mesh-generation/text-to-raw-mesh" \
  -H "Content-Type: application/json" \
  -d '{
    "text_prompt": "A red sports car",
    "output_format": "glb"
  }'

# Response: {"job_id": "job_123456", "status": "queued", "message": "..."}

# 2. Check job status
curl "http://localhost:8000/api/v1/system/jobs/job_123456"

# 3. Download result when completed
curl "http://localhost:8000/api/v1/system/jobs/job_123456/download" \
  -o "generated_car.glb"
```

### Example 2: Image to Mesh with Upload
```bash
# 1. Upload image
curl -X POST "http://localhost:8000/api/v1/mesh-generation/upload-image" \
  -F "file=@car_image.jpg"

# 2. Generate mesh from uploaded image
curl -X POST "http://localhost:8000/api/v1/mesh-generation/image-to-raw-mesh-upload" \
  -F "image_file=@car_image.jpg" \
  -F "output_format=glb"

# 3. Check status and download result
curl "http://localhost:8000/api/v1/system/jobs/{job_id}/download" -o "result.glb"
```

### Example 3: Mesh Segmentation
```bash
# 1. Segment mesh
curl -X POST "http://localhost:8000/api/v1/mesh-segmentation/segment-mesh" \
  -H "Content-Type: application/json" \
  -d '{
    "mesh_path": "/path/to/mesh.glb",
    "num_parts": 8,
    "output_format": "glb"
  }'

# 2. Download segmented result
curl "http://localhost:8000/api/v1/system/jobs/{job_id}/download" -o "segmented.glb"
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_VALUE` | Invalid input parameter value |
| `NOT_FOUND` | Resource not found |
| `API_ERROR` | General API error |
| `INTERNAL_ERROR` | Internal server error |
| `FILE_UPLOAD_ERROR` | File upload failed |
| `MODEL_NOT_AVAILABLE` | Requested model not available |
| `INSUFFICIENT_VRAM` | Not enough VRAM for operation |
| `JOB_FAILED` | Job processing failed |

---

## Model Preferences

### Available Models by Feature

| Feature | Model ID | Description |
|---------|----------|-------------|
| Text to Raw Mesh | `trellis_text_to_raw_mesh` | TRELLIS model for text-to-mesh |
| Text to Textured Mesh | `trellis_text_to_textured_mesh` | TRELLIS model for textured mesh |
| Image to Raw Mesh | `trellis_image_to_raw_mesh` | TRELLIS model for image-to-mesh |
| Image to Textured Mesh | `trellis_image_to_textured_mesh` | TRELLIS model for textured mesh |
| Image Mesh Painting | `trellis_image_mesh_painting` | TRELLIS model for mesh painting |
| Part Completion | `holopart_part_completion` | HoloPart model for part completion |
| Mesh Segmentation | `partfield_mesh_segmentation` | PartField model for segmentation |
| Auto Rigging | `unirig_auto_rig` | UniRig model for auto-rigging |

---

## File Format Support

### Input Formats
- **Text**: Plain text strings
- **Images**: PNG, JPG, JPEG, WebP, BMP, TIFF
- **Meshes**: OBJ, GLB, GLTF, PLY, STL, FBX
- **Base64**: Encoded images and meshes

### Output Formats
- **Meshes**: OBJ, GLB, PLY, FBX
- **Textures**: PNG, JPG
- **Downloads**: Direct file or Base64 encoded

### File Size Limits
- **Images**: 50MB maximum
- **Meshes**: 200MB maximum
- **Image Resolution**: Up to 4096x4096 pixels

---

## Support and Troubleshooting

### Common Issues

1. **Job Stuck in Queue**: Check system status and GPU availability
2. **Model Not Available**: Verify model preference and feature compatibility
3. **File Upload Failed**: Check file size and format requirements
4. **Generation Failed**: Check input parameters and system resources

### Getting Help

- Check the interactive API documentation at `/docs`
- Monitor system status at `/api/v1/system/status`
- Review job logs and error messages in responses
- Ensure adequate VRAM is available for model operations

---

*Last updated: [Current Date]*
*API Version: 1.0.0* 