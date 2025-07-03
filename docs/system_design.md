# 3D Generative Models Backend System Design

## Overview

This document describes the architecture of a scalable and extensible FastAPI server framework for 3D generative AI models. The system supports multiple features including mesh generation, texture generation, mesh segmentation, and auto-rigging.

## System Architecture

### 1. High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Web Frontend  │    │  Mobile Apps    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Load Balancer      │
                    │    (nginx/traefik)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    FastAPI Gateway      │
                    │   (Main Entry Point)    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼─────────┐ ┌─────▼─────┐ ┌─────────▼─────────┐
    │  Request Router   │ │ Auth/Rate │ │   Job Scheduler   │
    │    & Validator    │ │  Limiter  │ │    & Queue       │
    └─────────┬─────────┘ └───────────┘ └─────────┬─────────┘
              │                                   │
              └───────────────┬───────────────────┘
                              │
                 ┌────────────▼────────────┐
                 │   Model Worker Pool     │
                 │    (GPU Scheduling)     │
                 └────────────┬────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐   ┌────────▼────────┐
│  GPU Worker   │    │   GPU Worker    │   │   GPU Worker    │
│   (VRAM: 8GB) │    │  (VRAM: 24GB)   │   │  (VRAM: 40GB)   │
│   - Model A   │    │   - Model B     │   │   - Model C     │
│   - Model D   │    │   - Model E     │   │   - Model F     │
└───────────────┘    └─────────────────┘   └─────────────────┘
```

### 2. Core Components

#### 2.1 FastAPI Gateway (Main Entry Point)
- **Purpose**: Single entry point for all client requests
- **Responsibilities**:
  - Request routing and validation
  - Authentication and authorization
  - Rate limiting and throttling
  - API documentation generation
  - Error handling and logging

#### 2.2 Model Abstraction Layer
- **Purpose**: Abstract interface for all AI models
- **Components**:
  - Base model interface
  - Model metadata management
  - Feature-specific implementations
  - Model lifecycle management

#### 2.3 VRAM-Aware Scheduler
- **Purpose**: Intelligent GPU resource management
- **Features**:
  - Dynamic model loading/unloading
  - VRAM usage monitoring
  - Optimal GPU allocation
  - Queue priority management

### 3. Supported Features

#### 3.1 Text/Image Conditioned Mesh Generation
- **Input**: Text prompt OR Single/Multiple images
- **Output**: Textured 3D mesh (GLB/OBJ/FBX)
- **Models**: TRELLIS, Hunyuan3D-2, HoloPart
- **VRAM Requirements**: 8GB - 40GB

#### 3.2 Text/Image Conditioned Texture Generation
- **Input**: Text prompt OR Single/Multiple images + Base mesh
- **Output**: Textured 3D mesh
- **Models**: Hunyuan3D Paint, TRELLIS Texture
- **VRAM Requirements**: 6GB - 24GB

#### 3.3 Mesh Segmentation
- **Input**: GLB mesh
- **Output**: Segmented GLB mesh with part labels
- **Models**: PartField, HoloPart
- **VRAM Requirements**: 4GB - 16GB

#### 3.4 Auto-Rigging
- **Input**: OBJ/GLB/FBX mesh without armature
- **Output**: Rigged mesh with bone structure
- **Models**: UniRig
- **VRAM Requirements**: 8GB - 20GB

## Detailed Design

### 1. Model Abstraction Layer

#### 1.1 Base Model Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum

class ModelStatus(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    PROCESSING = "processing"
    ERROR = "error"

class BaseModel(ABC):
    def __init__(self, model_id: str, vram_requirement: int):
        self.model_id = model_id
        self.vram_requirement = vram_requirement  # in MB
        self.status = ModelStatus.UNLOADED
        self.gpu_id: Optional[int] = None
    
    @abstractmethod
    async def load(self, gpu_id: int) -> bool:
        """Load model on specified GPU"""
        pass
    
    @abstractmethod
    async def unload(self) -> bool:
        """Unload model from GPU"""
        pass
    
    @abstractmethod
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return results"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats"""
        pass
```

#### 1.2 Feature-Specific Implementations

```python
class MeshGenerationModel(BaseModel):
    def __init__(self, model_id: str, vram_requirement: int):
        super().__init__(model_id, vram_requirement)
        self.feature_type = "mesh_generation"
    
    @abstractmethod
    async def generate_mesh(
        self, 
        text_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        **kwargs
    ) -> str:  # Returns path to generated mesh
        pass

class TextureGenerationModel(BaseModel):
    def __init__(self, model_id: str, vram_requirement: int):
        super().__init__(model_id, vram_requirement)
        self.feature_type = "texture_generation"
    
    @abstractmethod
    async def generate_texture(
        self,
        mesh_path: str,
        text_prompt: Optional[str] = None,
        images: Optional[List[str]] = None,
        **kwargs
    ) -> str:  # Returns path to textured mesh
        pass
```

### 2. VRAM-Aware Scheduler

#### 2.1 GPU Resource Monitor

```python
import GPUtil
import psutil
from typing import Dict, List

class GPUMonitor:
    def __init__(self):
        self.gpus = GPUtil.getGPUs()
    
    def get_gpu_status(self) -> List[Dict]:
        """Get current status of all GPUs"""
        status = []
        for gpu in self.gpus:
            status.append({
                'id': gpu.id,
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'memory_free': gpu.memoryFree,
                'utilization': gpu.load,
                'temperature': gpu.temperature
            })
        return status
    
    def find_best_gpu(self, required_memory: int) -> Optional[int]:
        """Find GPU with enough free memory"""
        for gpu in self.gpus:
            if gpu.memoryFree >= required_memory:
                return gpu.id
        return None
```

#### 2.2 Model Scheduler

```python
import asyncio
from typing import Dict, List, Optional
from collections import deque

class ModelScheduler:
    def __init__(self, gpu_monitor: GPUMonitor):
        self.gpu_monitor = gpu_monitor
        self.loaded_models: Dict[int, List[BaseModel]] = {}  # gpu_id -> models
        self.job_queue = deque()
        self.processing_jobs: Dict[str, Dict] = {}
        
    async def schedule_job(self, job_request: Dict) -> str:
        """Schedule a new job"""
        job_id = generate_job_id()
        job_request['job_id'] = job_id
        job_request['status'] = 'queued'
        job_request['created_at'] = datetime.utcnow()
        
        self.job_queue.append(job_request)
        return job_id
    
    async def process_queue(self):
        """Main queue processing loop"""
        while True:
            if self.job_queue:
                job = self.job_queue.popleft()
                await self._process_job(job)
            await asyncio.sleep(0.1)
    
    async def _process_job(self, job: Dict):
        """Process a single job"""
        try:
            # Find suitable model
            model = self._find_model(job['feature'], job.get('model_preference'))
            
            # Ensure model is loaded
            await self._ensure_model_loaded(model, job['vram_requirement'])
            
            # Process request
            result = await model.process(job['inputs'])
            
            # Update job status
            job['status'] = 'completed'
            job['result'] = result
            
        except Exception as e:
            job['status'] = 'error'
            job['error'] = str(e)
        
        self.processing_jobs[job['job_id']] = job
```

### 3. FastAPI Server Structure

#### 3.1 Main Server Entry

```python
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_system()
    yield
    # Shutdown
    await cleanup_system()

app = FastAPI(
    title="3D Generative Models API",
    description="Scalable 3D AI model inference server",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include feature routers
app.include_router(mesh_generation_router, prefix="/api/v1/mesh", tags=["Mesh Generation"])
app.include_router(texture_generation_router, prefix="/api/v1/texture", tags=["Texture Generation"])
app.include_router(segmentation_router, prefix="/api/v1/segment", tags=["Mesh Segmentation"])
app.include_router(rigging_router, prefix="/api/v1/rig", tags=["Auto-Rigging"])
app.include_router(system_router, prefix="/api/v1/system", tags=["System"])
```

#### 3.2 Feature-Specific Routers

```python
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List

mesh_generation_router = APIRouter()

class MeshGenerationRequest(BaseModel):
    text_prompt: Optional[str] = None
    images: Optional[List[str]] = None  # Base64 encoded or URLs
    model_preference: Optional[str] = None
    output_format: str = "glb"
    quality: str = "medium"  # low, medium, high
    texture_resolution: int = 1024

@mesh_generation_router.post("/generate")
async def generate_mesh(
    request: MeshGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate 3D mesh from text or images"""
    if not request.text_prompt and not request.images:
        raise HTTPException(400, "Either text_prompt or images must be provided")
    
    job_id = await scheduler.schedule_job({
        'feature': 'mesh_generation',
        'inputs': request.dict(),
        'vram_requirement': get_model_vram_requirement(request.model_preference)
    })
    
    return {"job_id": job_id, "status": "queued"}

@mesh_generation_router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results"""
    job = scheduler.get_job_status(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job
```

### 4. Configuration Management

#### 4.1 Model Registry

```yaml
# config/models.yaml
models:
  mesh_generation:
    trellis_image_large:
      class: "TrellisImageModel"
      vram_requirement: 8192  # MB
      supported_inputs: ["image"]
      supported_outputs: ["glb", "obj"]
      max_concurrent: 2
    
    hunyuan3d_mini:
      class: "Hunyuan3DModel"
      vram_requirement: 6144
      supported_inputs: ["text", "image"]
      supported_outputs: ["glb", "obj", "fbx"]
      max_concurrent: 3
    
    trellis_text_xlarge:
      class: "TrellisTextModel"
      vram_requirement: 16384
      supported_inputs: ["text"]
      supported_outputs: ["glb", "obj"]
      max_concurrent: 1

  texture_generation:
    hunyuan3d_paint:
      class: "Hunyuan3DPaintModel"
      vram_requirement: 12288
      supported_inputs: ["mesh", "text", "image"]
      supported_outputs: ["glb", "obj"]
      max_concurrent: 2

  segmentation:
    partfield:
      class: "PartFieldModel"
      vram_requirement: 8192
      supported_inputs: ["glb"]
      supported_outputs: ["glb"]
      max_concurrent: 3

  auto_rigging:
    unirig:
      class: "UniRigModel"
      vram_requirement: 10240
      supported_inputs: ["obj", "glb", "fbx"]
      supported_outputs: ["obj", "glb", "fbx"]
      max_concurrent: 2
```

#### 4.2 System Configuration

```yaml
# config/system.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_request_size: 100MB
  timeout: 300

gpu:
  auto_detect: true
  memory_buffer: 1024  # MB to keep free
  max_utilization: 0.9

scheduler:
  queue_size: 1000
  job_timeout: 1800  # 30 minutes
  cleanup_interval: 300  # 5 minutes
  priority_weights:
    mesh_generation: 1.0
    texture_generation: 0.8
    segmentation: 0.6
    auto_rigging: 0.4

storage:
  input_dir: "/tmp/inputs"
  output_dir: "/tmp/outputs"
  cleanup_after: 3600  # 1 hour
  max_file_size: 500MB

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/server.log"
  max_size: 10MB
  backup_count: 5
```

### 5. Error Handling and Monitoring

#### 5.1 Error Response Format

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any

class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: str

# Common error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            timestamp=datetime.utcnow().isoformat(),
            request_id=request.headers.get("X-Request-ID", "unknown")
        ).dict()
    )
```

#### 5.2 Health Monitoring

```python
@system_router.get("/health")
async def health_check():
    """System health check"""
    gpu_status = gpu_monitor.get_gpu_status()
    queue_size = len(scheduler.job_queue)
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu_status": gpu_status,
        "queue_size": queue_size,
        "active_jobs": len(scheduler.processing_jobs)
    }

@system_router.get("/metrics")
async def get_metrics():
    """System metrics for monitoring"""
    return {
        "queue_metrics": scheduler.get_queue_metrics(),
        "gpu_metrics": gpu_monitor.get_detailed_metrics(),
        "model_metrics": model_registry.get_usage_metrics()
    }
```

## Security Considerations

### 1. Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- API key management for service-to-service communication

### 2. Input Validation
- File type validation
- Size limits enforcement
- Malicious content scanning

### 3. Rate Limiting
- Per-user request limits
- Feature-specific throttling
- Adaptive rate limiting based on system load

### 4. Data Protection
- Input/output encryption in transit
- Temporary file cleanup
- GDPR compliance for user data

## Scalability Features

### 1. Horizontal Scaling
- Kubernetes deployment support
- Load balancer integration
- Auto-scaling based on queue size

### 2. Distributed Computing
- Multi-node GPU clusters
- Model sharding for large models
- Distributed job queue (Redis/RabbitMQ)

### 3. Caching Strategy
- Model weight caching
- Result caching for common requests
- CDN integration for file serving

### 4. Performance Optimization
- Async/await throughout
- Connection pooling
- Background task processing
- Memory-efficient streaming

## Deployment Architecture

### 1. Container Strategy
```dockerfile
# Multi-stage build for optimization
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base
# ... base dependencies

FROM base as models
# ... model downloads and optimization

FROM base as runtime
COPY --from=models /models /app/models
# ... runtime setup
```

### 2. Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pocket3d-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pocket3d-api
  template:
    metadata:
      labels:
        app: pocket3d-api
    spec:
      containers:
      - name: api-server
        image: pocket3d:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
```

## Future Extensibility

### 1. Plugin Architecture
- Dynamic model loading
- Feature plugin system
- Third-party model integration

### 2. API Versioning
- Semantic versioning
- Backward compatibility
- Gradual migration support

### 3. Multi-Modal Support
- Video input processing
- Audio-driven generation
- Cross-modal conditioning

This system design provides a robust, scalable foundation for a 3D generative AI backend that can efficiently manage multiple models, optimize GPU resources, and handle high-throughput requests while maintaining extensibility for future enhancements.
