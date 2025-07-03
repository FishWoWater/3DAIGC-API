# Adapter Registration Guide

This guide explains how adapters are registered and used in the 3D Generative Models API.

## Overview

The adapter system follows this flow:

1. **Adapter Implementation** → 2. **Registry Configuration** → 3. **API Startup** → 4. **Automatic Registration** → 5. **API Endpoints**

## 1. Adapter Implementation

Adapters extend our base model classes and implement specific AI models:

```python
# Example: TRELLIS Text-to-Mesh Adapter
from core.models.mesh_models import TextToMeshModel

class TrellisTextToMeshAdapter(TextToMeshModel):
    def __init__(self, model_id="trellis-text", ...):
        super().__init__(model_id, ...)
    
    async def _load_model(self):
        # Load the actual TRELLIS model
        pass
    
    async def _process_request(self, inputs):
        # Process using TRELLIS
        pass
```

## 2. Registry Configuration

Adapters are configured in `config/adapter_registry.py`:

```python
class AdapterRegistry:
    def create_all_adapters(self):
        return [
            TrellisTextToMeshAdapter(
                model_id="trellis-text-to-mesh",
                vram_requirement=6144
            ),
            Hunyuan3DTextToMeshAdapter(
                model_id="hunyuan3d-text", 
                vram_requirement=8192
            ),
            # ... more adapters
        ]
```

## 3. API Startup Registration

In `api/main.py`, adapters are registered during startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize scheduler
    scheduler = ModelScheduler(gpu_monitor)
    
    # Register adapters
    adapter_registry = AdapterRegistry()
    models = adapter_registry.create_all_adapters()
    
    for model in models:
        scheduler.register_model(model)
    
    await scheduler.start()
    app.state.scheduler = scheduler
```

## 4. Using Adapters in API Endpoints

The registered adapters are automatically available through the scheduler:

### Text-to-Mesh Generation

```bash
curl -X POST "http://localhost:8000/api/v1/mesh/text-to-mesh" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "A detailed robot",
       "quality": "high",
       "output_format": "glb"
     }'
```

The system will:
1. Find the best available text-to-mesh adapter (TRELLIS, Hunyuan3D, etc.)
2. Load the model if needed
3. Process the request
4. Return job status

### Auto-Rigging

```bash
curl -X POST "http://localhost:8000/api/v1/auto-rig/generate-rig" \
     -H "Content-Type: application/json" \
     -d '{
       "mesh_path": "/path/to/mesh.obj",
       "rig_type": "humanoid",
       "output_format": "fbx"
     }'
```

### Mesh Segmentation

```bash
curl -X POST "http://localhost:8000/api/v1/mesh-segment/segment-mesh" \
     -H "Content-Type: application/json" \
     -d '{
       "mesh_path": "/path/to/mesh.glb",
       "num_segments": 8,
       "segmentation_method": "semantic"
     }'
```

## 5. Monitoring Adapters

### List All Registered Adapters

```bash
curl "http://localhost:8000/api/v1/system/adapters"
```

Response:
```json
{
  "adapters": [
    {
      "model_id": "trellis-text-to-mesh",
      "feature_type": "text_to_mesh",
      "status": "loaded",
      "vram_requirement": 6144,
      "processing_count": 0
    }
  ],
  "total_count": 4,
  "by_feature": {
    "text_to_mesh": ["trellis-text-to-mesh", "hunyuan3d-text"],
    "auto_rig": ["unirig-adapter"],
    "mesh_segmentation": ["partfield-segment"]
  }
}
```

### Check Scheduler Status

```bash
curl "http://localhost:8000/api/v1/system/scheduler-status"
```

### Monitor Job Progress

```bash
# Get job status
curl "http://localhost:8000/api/v1/system/jobs/{job_id}"

# Cancel a job
curl -X POST "http://localhost:8000/api/v1/system/jobs/{job_id}/cancel"
```

## 6. Adding New Adapters

### Step 1: Implement Adapter

```python
# adapters/my_new_adapter.py
from core.models.mesh_models import TextToMeshModel

class MyNewAdapter(TextToMeshModel):
    def __init__(self):
        super().__init__(
            model_id="my-new-model",
            model_path="/path/to/model",
            vram_requirement=4096,
            feature_type="text_to_mesh"
        )
    
    async def _load_model(self):
        # Your model loading code
        pass
    
    async def _process_request(self, inputs):
        # Your processing code
        return {"output_mesh_path": "generated.glb"}
```

### Step 2: Register in AdapterRegistry

```python
# config/adapter_registry.py
from adapters.my_new_adapter import MyNewAdapter

class AdapterRegistry:
    def create_all_adapters(self):
        return [
            # ... existing adapters
            MyNewAdapter(),
        ]
```

### Step 3: Restart API

The adapter will be automatically registered on next startup.

## 7. Configuration Options

### Environment Variables

```bash
# Model paths
export TRELLIS_MODEL_PATH="/path/to/trellis"
export HUNYUAN3D_MODEL_PATH="/path/to/hunyuan3d"

# VRAM settings
export DEFAULT_VRAM_BUFFER=1024

# Start server
python examples/api_with_adapters_demo.py
```

### YAML Configuration

```yaml
# config/models.yaml
adapters:
  trellis_text_to_mesh:
    enabled: true
    model_path: "/path/to/trellis"
    vram_requirement: 6144
  
  hunyuan3d_text_to_mesh:
    enabled: true
    model_path: "/path/to/hunyuan3d"
    vram_requirement: 8192
```

## 8. Error Handling

The system provides automatic error handling:

- **Model Loading Errors**: Models that fail to load are marked as "error" status
- **Processing Errors**: Failed jobs are marked as "failed" with error details
- **Resource Constraints**: Jobs wait in queue if insufficient VRAM
- **Capacity Limits**: New jobs wait if models are at max concurrent capacity

## 9. Performance Monitoring

### GPU Usage

```bash
curl "http://localhost:8000/api/v1/system/scheduler-status" | jq '.scheduler.gpu_status'
```

### Model Performance

```bash
curl "http://localhost:8000/api/v1/system/adapters" | jq '.by_status'
```

This registration system provides:
- ✅ Automatic adapter discovery
- ✅ Dynamic model loading/unloading
- ✅ VRAM-aware scheduling
- ✅ Concurrent processing limits
- ✅ Comprehensive monitoring
- ✅ Graceful error handling
