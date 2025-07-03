# API Reference Documentation

## Overview

The 3D Generative Models API provides endpoints for various 3D AI operations including mesh generation, texture generation, mesh segmentation, and auto-rigging. All endpoints return JSON responses and support both synchronous and asynchronous processing.

## Base URL
```
https://api.pocket3d.studio/api/v1
```

## Authentication

All API requests require authentication using either API keys or JWT tokens.

### API Key Authentication
```http
Authorization: Bearer YOUR_API_KEY
```

### JWT Token Authentication
```http
Authorization: Bearer YOUR_JWT_TOKEN
```

## Rate Limiting

- **Free Tier**: 10 requests per minute
- **Pro Tier**: 100 requests per minute  
- **Enterprise**: 1000 requests per minute

Rate limit headers are included in all responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1640995200
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error_code": "string",
  "message": "string", 
  "details": {},
  "timestamp": "2023-12-01T12:00:00Z",
  "request_id": "req_123abc"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_INPUT` | Invalid input parameters |
| `INSUFFICIENT_CREDITS` | Not enough API credits |
| `MODEL_UNAVAILABLE` | Requested model is not available |
| `PROCESSING_FAILED` | Model processing failed |
| `QUEUE_FULL` | Job queue is at capacity |
| `FILE_TOO_LARGE` | Uploaded file exceeds size limit |

## Mesh Generation

### Generate Mesh from Text

Generate a 3D mesh from a text description.

**Endpoint:** `POST /mesh/generate`

**Request Body:**
```json
{
  "text_prompt": "A red sports car",
  "model_preference": "trellis_text_xlarge",
  "output_format": "glb",
  "quality": "high",
  "texture_resolution": 2048,
  "seed": 42
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text_prompt` | string | Yes* | Text description of the 3D object |
| `images` | array[string] | Yes* | Base64 encoded images (alternative to text) |
| `model_preference` | string | No | Preferred model ID |
| `output_format` | string | No | Output format: `glb`, `obj`, `fbx` (default: `glb`) |
| `quality` | string | No | Quality level: `low`, `medium`, `high` (default: `medium`) |
| `texture_resolution` | integer | No | Texture resolution: 512, 1024, 2048, 4096 (default: 1024) |
| `seed` | integer | No | Random seed for reproducible results |

*Either `text_prompt` or `images` must be provided.

**Response:**
```json
{
  "job_id": "job_123abc",
  "status": "queued",
  "message": "Job queued successfully",
  "estimated_completion": "2023-12-01T12:05:00Z"
}
```

### Generate Mesh from Images

**Endpoint:** `POST /mesh/generate`

**Request Body:**
```json
{
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
  ],
  "model_preference": "trellis_image_large",
  "output_format": "glb"
}
```

### Upload Reference Images

Upload images to be referenced in mesh generation requests.

**Endpoint:** `POST /mesh/upload`

**Request:** Multipart form data with image files

**Response:**
```json
{
  "files": [
    "uploads/img_123abc.jpg",
    "uploads/img_456def.jpg"
  ],
  "message": "Uploaded 2 files"
}
```

### Get Job Status

**Endpoint:** `GET /mesh/status/{job_id}`

**Response:**
```json
{
  "job_id": "job_123abc",
  "status": "completed",
  "feature": "mesh_generation",
  "created_at": "2023-12-01T12:00:00Z",
  "started_at": "2023-12-01T12:01:00Z", 
  "completed_at": "2023-12-01T12:04:30Z",
  "result": {
    "file_path": "outputs/mesh_123abc.glb",
    "format": "glb",
    "file_size": 2048576,
    "vertices": 15420,
    "faces": 30840,
    "has_texture": true,
    "preview_image": "outputs/preview_123abc.jpg"
  }
}
```

**Status Values:**
- `queued` - Job is waiting in queue
- `processing` - Job is being processed
- `completed` - Job completed successfully
- `error` - Job failed with error

### Download Generated Mesh

**Endpoint:** `GET /mesh/download/{job_id}`

**Response:** Binary file download with appropriate content-type header.

### List Available Models

**Endpoint:** `GET /mesh/models`

**Response:**
```json
{
  "models": [
    {
      "model_id": "trellis_image_large",
      "status": "loaded",
      "gpu_id": 0,
      "vram_requirement": 8192,
      "processing_count": 0,
      "supported_formats": {
        "input": ["image"],
        "output": ["glb", "obj"]
      },
      "description": "TRELLIS large model for image-to-3D generation",
      "estimated_time": "3-5 minutes"
    }
  ]
}
```

## Texture Generation

### Generate Texture for Mesh

Apply textures to an existing 3D mesh based on text or image prompts.

**Endpoint:** `POST /texture/generate`

**Request Body:**
```json
{
  "mesh_file": "uploads/mesh_123.glb",
  "text_prompt": "Medieval stone castle walls",
  "images": ["data:image/jpeg;base64,..."],
  "model_preference": "hunyuan3d_paint",
  "texture_resolution": 2048,
  "style": "realistic"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mesh_file` | string | Yes | Path to uploaded mesh file |
| `text_prompt` | string | Yes* | Text description for texture |
| `images` | array[string] | Yes* | Reference images for texture style |
| `model_preference` | string | No | Preferred model ID |
| `texture_resolution` | integer | No | Texture resolution (default: 1024) |
| `style` | string | No | Style: `realistic`, `cartoon`, `artistic` |
| `intensity` | float | No | Texture intensity 0.0-1.0 (default: 0.8) |

*Either `text_prompt` or `images` must be provided.

**Response:**
```json
{
  "job_id": "job_texture_123",
  "status": "queued",
  "message": "Texture generation job queued"
}
```

## Mesh Segmentation

### Segment Mesh into Parts

Automatically segment a 3D mesh into semantic parts.

**Endpoint:** `POST /segment/analyze`

**Request Body:**
```json
{
  "mesh_file": "uploads/mesh_123.glb",
  "model_preference": "partfield",
  "segmentation_level": "coarse",
  "output_format": "glb"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mesh_file` | string | Yes | Path to mesh file to segment |
| `model_preference` | string | No | Preferred segmentation model |
| `segmentation_level` | string | No | `coarse`, `fine`, `semantic` (default: `coarse`) |
| `output_format` | string | No | Output format: `glb`, `obj` (default: `glb`) |
| `min_part_size` | integer | No | Minimum vertices per part (default: 100) |

**Response:**
```json
{
  "job_id": "job_segment_123",
  "status": "queued",
  "message": "Segmentation job queued"
}
```

**Completed Job Result:**
```json
{
  "result": {
    "file_path": "outputs/segmented_123.glb",
    "parts": [
      {
        "part_id": "head",
        "label": "Head",
        "vertices": 2450,
        "faces": 4900,
        "color": "#FF0000"
      },
      {
        "part_id": "body", 
        "label": "Body",
        "vertices": 5680,
        "faces": 11360,
        "color": "#00FF00"
      }
    ],
    "total_parts": 8
  }
}
```

## Auto-Rigging

### Generate Skeleton for Mesh

Automatically generate a bone structure and rig for character meshes.

**Endpoint:** `POST /rig/generate`

**Request Body:**
```json
{
  "mesh_file": "uploads/character_123.glb",
  "model_preference": "unirig",
  "character_type": "humanoid",
  "output_format": "glb"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mesh_file` | string | Yes | Path to character mesh file |
| `model_preference` | string | No | Preferred rigging model |
| `character_type` | string | No | `humanoid`, `quadruped`, `generic` (default: `humanoid`) |
| `output_format` | string | No | Output format: `glb`, `fbx`, `obj` (default: `glb`) |
| `bone_count` | integer | No | Target number of bones (default: auto) |
| `include_weights` | boolean | No | Include vertex weights (default: true) |

**Response:**
```json
{
  "job_id": "job_rig_123", 
  "status": "queued",
  "message": "Auto-rigging job queued"
}
```

**Completed Job Result:**
```json
{
  "result": {
    "file_path": "outputs/rigged_123.glb",
    "bone_count": 24,
    "bones": [
      {
        "name": "Root",
        "parent": null,
        "position": [0, 0, 0],
        "rotation": [0, 0, 0, 1]
      },
      {
        "name": "Spine",
        "parent": "Root", 
        "position": [0, 1.2, 0],
        "rotation": [0, 0, 0, 1]
      }
    ],
    "animation_ready": true
  }
}
```

## System Endpoints

### Health Check

**Endpoint:** `GET /system/health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T12:00:00Z",
  "gpu_status": [
    {
      "id": 0,
      "name": "NVIDIA RTX 4090",
      "memory_total": 24576,
      "memory_used": 8192,
      "memory_free": 16384,
      "utilization": 0.65,
      "temperature": 72
    }
  ],
  "queue_size": 3,
  "active_jobs": 2
}
```

### System Metrics

**Endpoint:** `GET /system/metrics`

**Response:**
```json
{
  "queue_metrics": {
    "total_jobs_processed": 1247,
    "average_processing_time": 185.4,
    "current_queue_size": 5,
    "jobs_by_status": {
      "queued": 5,
      "processing": 3,
      "completed": 1239,
      "error": 8
    }
  },
  "gpu_metrics": {
    "total_gpus": 2,
    "gpus_available": 2,
    "average_utilization": 0.73,
    "memory_usage": {
      "total": 49152,
      "used": 20480,
      "free": 28672
    }
  },
  "model_metrics": {
    "models_loaded": 6,
    "models_available": 12,
    "usage_stats": {
      "trellis_image_large": {
        "requests": 456,
        "avg_time": 180.2,
        "success_rate": 0.98
      }
    }
  }
}
```

### Worker Status

**Endpoint:** `GET /system/workers`

**Response:**
```json
{
  "workers": [
    {
      "worker_id": "worker_1",
      "gpu_id": 0,
      "status": "active",
      "current_job": "job_123abc",
      "models_loaded": ["trellis_image_large"],
      "uptime": 7200
    }
  ],
  "gpu_allocation": {
    "0": "image_worker",
    "1": "text_worker"
  }
}
```

## WebSocket Events

For real-time job progress updates, connect to the WebSocket endpoint:

**Endpoint:** `WSS /ws/jobs/{job_id}`

**Events:**

```json
// Job started
{
  "event": "job_started",
  "job_id": "job_123abc",
  "timestamp": "2023-12-01T12:01:00Z"
}

// Progress update
{
  "event": "progress_update", 
  "job_id": "job_123abc",
  "progress": 0.45,
  "stage": "generating_mesh",
  "eta": "2023-12-01T12:03:30Z"
}

// Job completed
{
  "event": "job_completed",
  "job_id": "job_123abc", 
  "result": {...}
}

// Job failed
{
  "event": "job_failed",
  "job_id": "job_123abc",
  "error": "Processing failed: insufficient memory"
}
```

## SDKs and Libraries

### Python SDK

```python
from pocket3d import Pocket3DClient

client = Pocket3DClient(api_key="your_api_key")

# Generate mesh from text
job = client.mesh.generate_from_text(
    prompt="A red sports car",
    quality="high"
)

# Wait for completion
result = client.wait_for_completion(job.job_id)
print(f"Mesh saved to: {result.file_path}")
```

### JavaScript SDK

```javascript
import { Pocket3DClient } from '@pocket3d/sdk';

const client = new Pocket3DClient({ apiKey: 'your_api_key' });

// Generate mesh from images
const job = await client.mesh.generateFromImages({
  images: ['data:image/jpeg;base64,...'],
  quality: 'medium'
});

// Listen for updates
client.on('progress', (data) => {
  console.log(`Progress: ${data.progress * 100}%`);
});

const result = await client.waitForCompletion(job.jobId);
```

## Best Practices

### 1. Efficient API Usage

- **Batch requests** when possible to reduce overhead
- **Use webhooks** for long-running jobs instead of polling
- **Cache results** to avoid duplicate processing
- **Compress images** before uploading to reduce transfer time

### 2. Error Handling

```python
try:
    job = client.mesh.generate_from_text("A blue car")
    result = client.wait_for_completion(job.job_id, timeout=300)
except Pocket3DAPIError as e:
    if e.error_code == "INSUFFICIENT_CREDITS":
        # Handle billing issue
        pass
    elif e.error_code == "MODEL_UNAVAILABLE":
        # Try with different model
        pass
    else:
        # Log error for investigation
        logger.error(f"Unexpected error: {e}")
```

### 3. Performance Optimization

- **Pre-upload large files** using the upload endpoints
- **Specify model preferences** to avoid automatic selection overhead  
- **Use appropriate quality settings** - higher quality takes longer
- **Monitor your quota** to avoid rate limiting

### 4. File Management

- **Clean up temporary files** after processing
- **Use signed URLs** for secure file access
- **Implement retry logic** for failed downloads
- **Validate file formats** before uploading

## Webhooks

Configure webhooks to receive notifications when jobs complete:

**Webhook URL Configuration:**
```json
{
  "url": "https://your-app.com/webhooks/pocket3d",
  "events": ["job_completed", "job_failed"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload:**
```json
{
  "event": "job_completed",
  "job_id": "job_123abc",
  "timestamp": "2023-12-01T12:04:30Z",
  "data": {
    "feature": "mesh_generation",
    "result": {
      "file_path": "outputs/mesh_123abc.glb",
      "download_url": "https://api.pocket3d.studio/download/mesh_123abc.glb?token=..."
    }
  }
}
```

## Limits and Quotas

### File Limits
- **Maximum file size**: 500MB per file
- **Maximum files per request**: 10 files
- **Supported formats**: JPEG, PNG, WebP for images; GLB, OBJ, FBX for meshes

### Processing Limits
- **Maximum processing time**: 30 minutes per job
- **Maximum concurrent jobs**: Varies by tier
- **Queue limit**: 1000 jobs per account

### Rate Limits
- **API calls**: Based on subscription tier
- **File uploads**: 100MB per minute
- **WebSocket connections**: 10 concurrent connections

This API reference provides comprehensive documentation for all endpoints and features of the 3D Generative Models backend system.
