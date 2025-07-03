# Implementation Guide for 3D Generative Models Backend

## Quick Start

### 1. Project Structure
```
backend/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── dependencies.py         # Dependency injection
│   └── routers/
│       ├── __init__.py
│       ├── mesh_generation.py  # Mesh generation endpoints
│       ├── texture_generation.py
│       ├── segmentation.py
│       ├── auto_rigging.py
│       └── system.py          # Health, metrics endpoints
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── models/                # Model abstractions
│   │   ├── __init__.py
│   │   ├── base.py           # Base model interface
│   │   ├── mesh_models.py    # Mesh generation models
│   │   ├── texture_models.py
│   │   ├── segment_models.py
│   │   └── rig_models.py
│   ├── scheduler/             # VRAM-aware scheduler
│   │   ├── __init__.py
│   │   ├── gpu_monitor.py
│   │   ├── model_scheduler.py
│   │   └── job_queue.py
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py
│       ├── validation.py
│       └── exceptions.py
├── adapters/                  # Third-party model adapters
│   ├── __init__.py
│   ├── trellis_adapter.py
│   ├── hunyuan3d_adapter.py
│   ├── partfield_adapter.py
│   ├── holopart_adapter.py
│   └── unirig_adapter.py
├── config/
│   ├── models.yaml           # Model configurations
│   ├── system.yaml          # System configurations
│   └── logging.yaml         # Logging configurations
├── tests/
│   ├── test_models/
│   ├── test_scheduler/
│   └── test_api/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
└── scripts/
    ├── setup.sh
    ├── download_models.sh
    └── run_server.sh
```

### 2. Installation Steps

#### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install CUDA dependencies (if using GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 2: Configuration
```bash
# Copy configuration templates
cp config/models.yaml.template config/models.yaml
cp config/system.yaml.template config/system.yaml

# Edit configurations as needed
nano config/system.yaml
```

#### Step 3: Model Downloads
```bash
# Download required models
./scripts/download_models.sh

# Verify model installations
python -c "from core.models import verify_models; verify_models()"
```

#### Step 4: Start Server
```bash
# Development mode
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
./scripts/run_server.sh
```

## Implementation Priority

### Phase 1: Core Infrastructure (Week 1-2)
1. **Base Model Interface**
   - Implement `BaseModel` abstract class
   - Create model registry system
   - Basic GPU monitoring

2. **FastAPI Framework**
   - Main application setup
   - Basic routing structure
   - Error handling middleware

3. **Configuration System**
   - YAML-based configuration
   - Environment variable support
   - Validation schemas

### Phase 2: Basic Scheduler (Week 3-4)
1. **GPU Resource Management**
   - GPU detection and monitoring
   - Memory usage tracking
   - Basic allocation logic

2. **Job Queue System**
   - Simple FIFO queue
   - Job status tracking
   - Basic job processing

3. **Model Loading/Unloading**
   - Dynamic model loading
   - Memory management
   - Error recovery

### Phase 3: Model Adapters (Week 5-8)
1. **Mesh Generation Models**
   - TRELLIS adapter
   - Hunyuan3D adapter
   - Input/output standardization

2. **Additional Features**
   - Texture generation
   - Mesh segmentation
   - Auto-rigging

3. **Testing & Validation**
   - Unit tests for each adapter
   - Integration tests
   - Performance benchmarks

### Phase 4: Advanced Features (Week 9-12)
1. **Advanced Scheduling**
   - Priority queues
   - VRAM optimization
   - Load balancing

2. **Monitoring & Metrics**
   - Prometheus metrics
   - Health checks
   - Performance monitoring

3. **Security & Authentication**
   - JWT authentication
   - Rate limiting
   - Input validation

## Code Templates

### 1. Base Model Implementation

```python
# core/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import asyncio
import torch
from pathlib import Path

class ModelStatus(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    PROCESSING = "processing"
    ERROR = "error"

class BaseModel(ABC):
    """Base class for all AI models"""
    
    def __init__(
        self,
        model_id: str,
        model_path: str,
        vram_requirement: int,
    ):
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.vram_requirement = vram_requirement  # MB
        self.status = ModelStatus.UNLOADED
        self.gpu_id: Optional[int] = None
        self.model = None
        self.load_lock = asyncio.Lock()
        self.processing_count = 0
    
    @abstractmethod
    async def _load_model(self) -> Any:
        """Load the actual model. Override in subclasses."""
        pass
    
    @abstractmethod
    async def _unload_model(self) -> None:
        """Unload the actual model. Override in subclasses."""
        pass
    
    @abstractmethod
    async def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single request. Override in subclasses."""
        pass
    
    async def load(self, gpu_id: int) -> bool:
        """Load model on specified GPU"""
        async with self.load_lock:
            if self.status == ModelStatus.LOADED:
                return True
            
            try:
                self.status = ModelStatus.LOADING
                self.gpu_id = gpu_id
                
                # Set CUDA device
                if torch.cuda.is_available():
                    torch.cuda.set_device(gpu_id)
                
                # Load model
                self.model = await self._load_model()
                self.status = ModelStatus.LOADED
                return True
                
            except Exception as e:
                self.status = ModelStatus.ERROR
                raise Exception(f"Failed to load model {self.model_id}: {str(e)}")
    
    async def unload(self) -> bool:
        """Unload model from GPU"""
        async with self.load_lock:
            if self.status == ModelStatus.UNLOADED:
                return True
            
            try:
                await self._unload_model()
                self.model = None
                self.status = ModelStatus.UNLOADED
                self.gpu_id = None
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
                
            except Exception as e:
                self.status = ModelStatus.ERROR
                raise Exception(f"Failed to unload model {self.model_id}: {str(e)}")
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return results"""
        if self.status != ModelStatus.LOADED:
            raise Exception(f"Model {self.model_id} is not loaded")
        
        
        try:
            self.processing_count += 1
            self.status = ModelStatus.PROCESSING
            
            result = await self._process_request(inputs)
            return result
            
        finally:
            self.processing_count -= 1
            if self.processing_count == 0:
                self.status = ModelStatus.LOADED
    
    @abstractmethod
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Return supported input/output formats"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "gpu_id": self.gpu_id,
            "vram_requirement": self.vram_requirement,
            "processing_count": self.processing_count,
            "supported_formats": self.get_supported_formats()
        }
```

### 2. GPU Monitor Implementation

```python
# core/scheduler/gpu_monitor.py
import GPUtil
import psutil
import torch
from typing import Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class GPUMonitor:
    """Monitor GPU resources and usage"""
    
    def __init__(self, memory_buffer: int = 1024):
        self.memory_buffer = memory_buffer  # MB to keep free
        self.gpus = []
        self._update_gpu_list()
    
    def _update_gpu_list(self):
        """Update list of available GPUs"""
        if torch.cuda.is_available():
            self.gpus = GPUtil.getGPUs()
        else:
            self.gpus = []
            logger.warning("CUDA not available, running in CPU mode")
    
    def get_gpu_status(self) -> List[Dict]:
        """Get current status of all GPUs"""
        self._update_gpu_list()
        status = []
        
        for gpu in self.gpus:
            status.append({
                'id': gpu.id,
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'memory_free': gpu.memoryFree,
                'memory_utilization': gpu.memoryUtil,
                'gpu_utilization': gpu.load,
                'temperature': gpu.temperature
            })
        
        return status
    
    def find_best_gpu(self, required_memory: int) -> Optional[int]:
        """Find GPU with enough free memory"""
        self._update_gpu_list()
        
        best_gpu = None
        max_free_memory = 0
        
        for gpu in self.gpus:
            available_memory = gpu.memoryFree - self.memory_buffer
            
            if available_memory >= required_memory and available_memory > max_free_memory:
                best_gpu = gpu.id
                max_free_memory = available_memory
        
        return best_gpu
    
    def get_gpu_memory_usage(self, gpu_id: int) -> Dict:
        """Get memory usage for specific GPU"""
        if gpu_id >= len(self.gpus):
            raise ValueError(f"GPU {gpu_id} not found")
        
        gpu = self.gpus[gpu_id]
        return {
            'total': gpu.memoryTotal,
            'used': gpu.memoryUsed,
            'free': gpu.memoryFree,
            'utilization': gpu.memoryUtil
        }
    
    async def monitor_continuously(self, interval: int = 5):
        """Continuously monitor GPU status"""
        while True:
            try:
                status = self.get_gpu_status()
                logger.debug(f"GPU Status: {status}")
                
                # Check for overheating or high utilization
                for gpu_info in status:
                    if gpu_info['temperature'] > 85:
                        logger.warning(f"GPU {gpu_info['id']} temperature high: {gpu_info['temperature']}°C")
                    
                    if gpu_info['gpu_utilization'] > 0.95:
                        logger.warning(f"GPU {gpu_info['id']} utilization high: {gpu_info['gpu_utilization']:.1%}")
                
            except Exception as e:
                logger.error(f"Error monitoring GPUs: {str(e)}")
            
            await asyncio.sleep(interval)
```

### 3. Model Scheduler Implementation

```python
# core/scheduler/model_scheduler.py
import asyncio
from typing import Dict, List, Optional, Set
from collections import deque
import logging
import uuid
from datetime import datetime, timedelta
from core.models.base import BaseModel, ModelStatus
from core.scheduler.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)

class JobRequest:
    def __init__(
        self,
        feature: str,
        inputs: Dict,
        model_preference: Optional[str] = None,
        priority: int = 1
    ):
        self.job_id = str(uuid.uuid4())
        self.feature = feature
        self.inputs = inputs
        self.model_preference = model_preference
        self.priority = priority
        self.status = "queued"
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None

class ModelScheduler:
    """VRAM-aware model scheduler"""
    
    def __init__(self, gpu_monitor: GPUMonitor):
        self.gpu_monitor = gpu_monitor
        self.models: Dict[str, BaseModel] = {}
        self.loaded_models: Dict[int, Set[str]] = {}  # gpu_id -> model_ids
        self.job_queue = deque()
        self.processing_jobs: Dict[str, JobRequest] = {}
        self.completed_jobs: Dict[str, JobRequest] = {}
        self.max_completed_jobs = 1000
        self.running = False
        
    def register_model(self, model: BaseModel):
        """Register a model with the scheduler"""
        self.models[model.model_id] = model
        logger.info(f"Registered model: {model.model_id}")
    
    async def schedule_job(self, job_request: JobRequest) -> str:
        """Schedule a new job"""
        if len(self.job_queue) > 1000:  # Prevent queue overflow
            raise Exception("Job queue is full")
        
        self.job_queue.append(job_request)
        logger.info(f"Scheduled job {job_request.job_id} for feature {job_request.feature}")
        return job_request.job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status and results"""
        # Check processing jobs
        if job_id in self.processing_jobs:
            job = self.processing_jobs[job_id]
            return self._job_to_dict(job)
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return self._job_to_dict(job)
        
        # Check queued jobs
        for job in self.job_queue:
            if job.job_id == job_id:
                return self._job_to_dict(job)
        
        return None
    
    def _job_to_dict(self, job: JobRequest) -> Dict:
        """Convert job to dictionary"""
        return {
            "job_id": job.job_id,
            "feature": job.feature,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "result": job.result,
            "error": job.error
        }
    
    async def start(self):
        """Start the scheduler"""
        self.running = True
        logger.info("Starting model scheduler")
        
        # Start background tasks
        await asyncio.gather(
            self._process_queue(),
            self._cleanup_completed_jobs(),
            self.gpu_monitor.monitor_continuously()
        )
    
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("Stopping model scheduler")
    
    async def _process_queue(self):
        """Main queue processing loop"""
        while self.running:
            try:
                if self.job_queue:
                    job = self.job_queue.popleft()
                    await self._process_job(job)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in queue processing: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_job(self, job: JobRequest):
        """Process a single job"""
        try:
            job.status = "processing"
            job.started_at = datetime.utcnow()
            self.processing_jobs[job.job_id] = job
            
            # Find suitable model
            model = self._find_best_model(job.feature, job.model_preference)
            if not model:
                raise Exception(f"No suitable model found for feature: {job.feature}")
            
            # Ensure model is loaded
            await self._ensure_model_loaded(model)
            
            # Process request
            result = await model.process(job.inputs)
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.result = result
            
            logger.info(f"Completed job {job.job_id} using model {model.model_id}")
            
        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Error processing job {job.job_id}: {str(e)}")
        
        finally:
            # Move to completed jobs
            if job.job_id in self.processing_jobs:
                del self.processing_jobs[job.job_id]
            self.completed_jobs[job.job_id] = job
    
    def _find_best_model(self, feature: str, preference: Optional[str] = None) -> Optional[BaseModel]:
        """Find the best model for a feature"""
        # Filter models by feature
        suitable_models = [
            model for model in self.models.values()
            if hasattr(model, 'feature_type') and model.feature_type == feature
        ]
        
        if not suitable_models:
            return None
        
        # If preference specified, try to find it
        if preference:
            for model in suitable_models:
                if model.model_id == preference:
                    return model
        
        # Otherwise, choose based on availability and VRAM
        suitable_models.sort(key=lambda m: (
            m.status != ModelStatus.LOADED,  # Prefer already loaded models
            m.vram_requirement,  # Prefer smaller models
            m.processing_count   # Prefer less busy models
        ))
        
        return suitable_models[0]
    
    async def _ensure_model_loaded(self, model: BaseModel):
        """Ensure model is loaded on a suitable GPU"""
        if model.status == ModelStatus.LOADED:
            return
        
        # Find suitable GPU
        gpu_id = self.gpu_monitor.find_best_gpu(model.vram_requirement)
        if gpu_id is None:
            # Try to free up space by unloading other models
            await self._free_gpu_space(model.vram_requirement)
            gpu_id = self.gpu_monitor.find_best_gpu(model.vram_requirement)
            
            if gpu_id is None:
                raise Exception("No GPU has enough memory for this model")
        
        # Load model
        await model.load(gpu_id)
        
        # Track loaded model
        if gpu_id not in self.loaded_models:
            self.loaded_models[gpu_id] = set()
        self.loaded_models[gpu_id].add(model.model_id)
    
    async def _free_gpu_space(self, required_memory: int):
        """Free up GPU space by unloading unused models"""
        for gpu_id, model_ids in self.loaded_models.items():
            gpu_memory = self.gpu_monitor.get_gpu_memory_usage(gpu_id)
            
            if gpu_memory['free'] >= required_memory:
                continue
            
            # Unload models that aren't currently processing
            for model_id in list(model_ids):
                model = self.models[model_id]
                if model.processing_count == 0:
                    await model.unload()
                    model_ids.remove(model_id)
                    logger.info(f"Unloaded model {model_id} from GPU {gpu_id}")
                    
                    # Check if we have enough space now
                    gpu_memory = self.gpu_monitor.get_gpu_memory_usage(gpu_id)
                    if gpu_memory['free'] >= required_memory:
                        return
    
    async def _cleanup_completed_jobs(self):
        """Clean up old completed jobs"""
        while self.running:
            try:
                if len(self.completed_jobs) > self.max_completed_jobs:
                    # Remove oldest jobs
                    oldest_jobs = sorted(
                        self.completed_jobs.values(),
                        key=lambda job: job.completed_at or datetime.utcnow()
                    )
                    
                    for job in oldest_jobs[:100]:  # Remove 100 oldest
                        del self.completed_jobs[job.job_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup: {str(e)}")
                await asyncio.sleep(60)
```

### 4. FastAPI Router Template

```python
# api/routers/mesh_generation.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from typing import Optional, List
import base64
import tempfile
import os
from pathlib import Path
import asyncio

from core.scheduler.model_scheduler import ModelScheduler, JobRequest
from core.utils.validation import validate_image, validate_text
from core.utils.file_utils import save_upload_file, cleanup_temp_files
from api.dependencies import get_scheduler

router = APIRouter()

class MeshGenerationRequest(BaseModel):
    text_prompt: Optional[str] = None
    images: Optional[List[str]] = None  # Base64 encoded images
    model_preference: Optional[str] = None
    output_format: str = "glb"
    quality: str = "medium"
    texture_resolution: int = 1024
    seed: Optional[int] = None
    
    @field_validator('output_format')
    def validate_output_format(cls, v):
        allowed_formats = ['glb', 'obj', 'fbx']
        if v not in allowed_formats:
            raise ValueError(f'Output format must be one of: {allowed_formats}')
        return v
    
    @field_validator('quality')
    def validate_quality(cls, v):
        allowed_qualities = ['low', 'medium', 'high']
        if v not in allowed_qualities:
            raise ValueError(f'Quality must be one of: {allowed_qualities}')
        return v
    
    @field_validator('texture_resolution')
    def validate_texture_resolution(cls, v):
        allowed_resolutions = [512, 1024, 2048, 4096]
        if v not in allowed_resolutions:
            raise ValueError(f'Texture resolution must be one of: {allowed_resolutions}')
        return v

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

@router.post("/generate", response_model=JobResponse)
async def generate_mesh(
    request: MeshGenerationRequest,
    scheduler: ModelScheduler = Depends(get_scheduler)
):
    """Generate 3D mesh from text or images"""
    
    # Validation
    if not request.text_prompt and not request.images:
        raise HTTPException(
            status_code=400, 
            detail="Either text_prompt or images must be provided"
        )
    
    if request.text_prompt:
        validate_text(request.text_prompt)
    
    if request.images:
        for img_b64 in request.images:
            validate_image(img_b64)
    
    try:
        # Create job request
        job_request = JobRequest(
            feature="mesh_generation",
            inputs=request.dict(),
            model_preference=request.model_preference,
            priority=1
        )
        
        # Schedule job
        job_id = await scheduler.schedule_job(job_request)
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            message="Job queued successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    scheduler: ModelScheduler = Depends(get_scheduler)
):
    """Get job status and results"""
    
    job = await scheduler.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job

@router.get("/download/{job_id}")
async def download_result(
    job_id: str,
    scheduler: ModelScheduler = Depends(get_scheduler)
):
    """Download generated mesh file"""
    
    job = await scheduler.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job['result'] or 'file_path' not in job['result']:
        raise HTTPException(status_code=404, detail="Result file not found")
    
    file_path = job['result']['file_path']
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        path=file_path,
        filename=f"mesh_{job_id}.{job['result'].get('format', 'glb')}",
        media_type='application/octet-stream'
    )

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    scheduler: ModelScheduler = Depends(get_scheduler)
):
    """Upload reference images for mesh generation"""
    
    if len(files) > 10:  # Limit number of files
        raise HTTPException(status_code=400, detail="Too many files")
    
    saved_files = []
    try:
        for file in files:
            # Validate file
            if file.content_type not in ['image/jpeg', 'image/png', 'image/webp']:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
            
            # Save file
            file_path = await save_upload_file(file)
            saved_files.append(file_path)
        
        return {"files": saved_files, "message": f"Uploaded {len(saved_files)} files"}
        
    except Exception as e:
        # Cleanup on error
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_available_models(
    scheduler: ModelScheduler = Depends(get_scheduler)
):
    """List available mesh generation models"""
    
    mesh_models = []
    for model_id, model in scheduler.models.items():
        if hasattr(model, 'feature_type') and model.feature_type == 'mesh_generation':
            mesh_models.append(model.get_info())
    
    return {"models": mesh_models}
```

## Testing Strategy

### 1. Unit Tests
```python
# tests/test_models/test_base_model.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from core.models.base import BaseModel, ModelStatus

class TestBaseModel(BaseModel):
    async def _load_model(self):
        return Mock()
    
    async def _unload_model(self):
        pass
    
    async def _process_request(self, inputs):
        return {"result": "test"}
    
    def get_supported_formats(self):
        return {"input": ["image"], "output": ["glb"]}

@pytest.mark.asyncio
async def test_model_loading():
    model = TestBaseModel("test_model", "/path/to/model", 1024)
    
    assert model.status == ModelStatus.UNLOADED
    
    success = await model.load(0)
    assert success
    assert model.status == ModelStatus.LOADED
    assert model.gpu_id == 0

@pytest.mark.asyncio
async def test_model_processing():
    model = TestBaseModel("test_model", "/path/to/model", 1024)
    await model.load(0)
    
    result = await model.process({"input": "test"})
    assert result == {"result": "test"}

@pytest.mark.asyncio
async def test_model_unloading():
    model = TestBaseModel("test_model", "/path/to/model", 1024)
    await model.load(0)
    
    success = await model.unload()
    assert success
    assert model.status == ModelStatus.UNLOADED
    assert model.gpu_id is None
```

### 2. Integration Tests
```python
# tests/test_api/test_mesh_generation.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_generate_mesh_with_text():
    response = client.post(
        "/api/v1/mesh/generate",
        json={
            "text_prompt": "A red car",
            "output_format": "glb",
            "quality": "medium"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"

def test_generate_mesh_validation_error():
    response = client.post(
        "/api/v1/mesh/generate",
        json={
            "output_format": "invalid_format"
        }
    )
    assert response.status_code == 422  # Validation error

def test_job_status():
    # First create a job
    response = client.post(
        "/api/v1/mesh/generate",
        json={"text_prompt": "A blue sphere"}
    )
    job_id = response.json()["job_id"]
    
    # Check status
    response = client.get(f"/api/v1/mesh/status/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
```

This implementation guide provides a comprehensive roadmap for building the 3D generative models backend. The templates show how to implement each component properly, with proper error handling, async/await patterns, and testing strategies.
