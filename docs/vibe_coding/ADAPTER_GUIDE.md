# ADAPTER IMPLEMENTATION AND USAGE GUIDE

## 🎯 Overview

This guide shows you how to implement and use model adapters in the Pocket3DStudio framework. Adapters allow you to integrate any AI model into our standardized API and scheduler system.

## 📋 Table of Contents

1. [Adapter Architecture](#adapter-architecture)
2. [Implementing Your Own Adapter](#implementing-your-own-adapter)
3. [Using Existing Adapters](#using-existing-adapters)
4. [Configuration and Deployment](#configuration-and-deployment)
5. [Testing and Debugging](#testing-and-debugging)
6. [Best Practices](#best-practices)

## 🏗️ Adapter Architecture

### What is an Adapter?

An adapter is a bridge between:
- **Your AI Model**: The actual implementation (TRELLIS, Hunyuan3D, etc.)
- **Our Framework**: Standardized interfaces, scheduling, and APIs

### Adapter Components

```
YourAdapter
├── Inherits from: BaseModel subclass (TextToMeshModel, etc.)
├── Implements: _load_model(), _unload_model(), _process_request()
├── Manages: Model lifecycle, VRAM, and processing
└── Returns: Standardized response format
```

## 🔧 Implementing Your Own Adapter

### Step 1: Choose the Right Base Class

```python
# For text-to-mesh generation
from core.models.mesh_models import TextToMeshModel

# For image-to-mesh generation  
from core.models.mesh_models import ImageToMeshModel

# For multimodal mesh generation
from core.models.mesh_models import MultiModalMeshModel

# For texture generation
from core.models.texture_models import TextToTextureModel

# For auto-rigging
from core.models.rig_models import AutoRigModel

# For mesh segmentation
from core.models.segment_models import MeshSegmentationModel
```

### Step 2: Create Your Adapter Class

```python
# filepath: adapters/my_model_adapter.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from core.models.mesh_models import TextToMeshModel

logger = logging.getLogger(__name__)


class MyModelAdapter(TextToMeshModel):
    """Adapter for My Custom Model."""
    
    def __init__(
        self,
        model_id: str = "my-model",
        model_path: Optional[str] = None,
        vram_requirement: int = 4096,  # MB
        max_concurrent: int = 1,
        custom_param: str = "default"
    ):
        # Set default paths
        if model_path is None:
            model_path = "/path/to/my/model"
        
        super().__init__(
            model_id=model_id,
            model_path=model_path,
            vram_requirement=vram_requirement,
            max_concurrent=max_concurrent,
            supported_output_formats=["glb", "obj"]  # What your model can output
        )
        
        # Store custom parameters
        self.custom_param = custom_param
        self.model_pipeline = None
    
    async def _load_model(self):
        """Load your model. Called once when model is needed."""
        try:
            logger.info(f"Loading my model: {self.model_id}")
            
            # Import your model's modules
            from my_model import MyModelPipeline
            
            # Initialize your model
            self.model_pipeline = MyModelPipeline.from_pretrained(
                self.model_path,
                device="cuda",
                custom_param=self.custom_param
            )
            
            logger.info("My model loaded successfully")
            return self.model_pipeline
            
        except Exception as e:
            logger.error(f"Failed to load my model: {str(e)}")
            raise Exception(f"Failed to load my model: {str(e)}")
    
    async def _unload_model(self):
        """Unload your model. Called when freeing VRAM."""
        try:
            if self.model_pipeline is not None:
                # Move to CPU and cleanup
                self.model_pipeline.cpu()
                del self.model_pipeline
                self.model_pipeline = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("My model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading my model: {str(e)}")
    
    async def _process_request(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a generation request.
        
        Args:
            inputs: Dictionary with validated inputs from API
            
        Returns:
            Dictionary with generation results
        """
        try:
            # Validate inputs using parent class
            output_format = self._validate_common_inputs(inputs)
            
            # Extract your model's parameters
            text_prompt = inputs["text_prompt"]
            quality = inputs.get("quality", "medium")
            
            logger.info(f"Generating with my model: '{text_prompt}'")
            
            # Run your model
            result = self.model_pipeline.generate(
                prompt=text_prompt,
                quality=quality
            )
            
            # Save output
            output_path = self._generate_output_path(text_prompt, output_format)
            result.save(output_path)
            
            # Create standardized response
            response = self._create_common_response(inputs, output_format)
            response.update({
                "output_mesh_path": str(output_path),
                "generation_info": {
                    "model": "MyModel",
                    "prompt": text_prompt,
                    "vertex_count": result.vertex_count,
                    "face_count": result.face_count
                }
            })
            
            logger.info(f"Generation completed: {output_path}")
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise Exception(f"Generation failed: {str(e)}")
    
    def _generate_output_path(self, prompt: str, format: str) -> Path:
        """Generate unique output file path."""
        # Create safe filename
        safe_name = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        # Create output directory
        output_dir = Path("/data/workspace/3DAIGC/Pocket3DStudio/backend/outputs/meshes")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        import time
        timestamp = int(time.time())
        filename = f"mymodel_{safe_name}_{timestamp}.{format}"
        
        return output_dir / filename
```

### Step 3: Register Your Adapter

```python
# In config/adapter_registry.py or your initialization code

from adapters.my_model_adapter import MyModelAdapter

# Register the adapter
adapter_registry.register_adapter(
    "my-model-adapter",
    MyModelAdapter, 
    "text_to_mesh"
)
```

### Step 4: Use Your Adapter

```python
# Create an instance
my_adapter = adapter_registry.create_adapter_instance(
    "my-model-adapter",
    vram_requirement=6144,
    max_concurrent=1,
    custom_param="my_value"
)

# Register with scheduler
scheduler.register_model(my_adapter)

# Use via API
job = JobRequest(
    feature="text_to_mesh",
    inputs={
        "text_prompt": "A beautiful 3D model",
        "quality": "high",
        "output_format": "glb"
    }
)
job_id = await scheduler.schedule_job(job)
```

## 🚀 Using Existing Adapters

### Available Adapters

1. **TRELLIS** - Text-to-mesh generation
2. **Hunyuan3D** - Text-to-mesh and image-to-mesh
3. **UniRig** - Automatic rigging
4. **PartField** - Mesh segmentation

### Quick Start

```python
from config.adapter_registry import register_all_adapters, adapter_registry

# Register all available adapters
register_all_adapters()

# Create specific adapter
trellis = adapter_registry.create_adapter_instance(
    "trellis-text-to-mesh",
    vram_requirement=6144
)

# Use with scheduler
scheduler.register_model(trellis)
```

### Configuration Options

```python
# TRELLIS Configuration
trellis_config = {
    "model_id": "trellis-custom",
    "vram_requirement": 8192,  # Higher for better quality
    "max_concurrent": 1,
    "trellis_root": "/path/to/TRELLIS",
    "model_path": "/path/to/pretrained"
}

# Hunyuan3D Configuration  
hunyuan_config = {
    "model_id": "hunyuan3d-high",
    "vram_requirement": 12288,  # 12GB for high quality
    "max_concurrent": 1,
    "hunyuan3d_root": "/path/to/Hunyuan3D-2"
}

# UniRig Configuration
unirig_config = {
    "model_id": "unirig-fast",
    "vram_requirement": 4096,
    "max_concurrent": 2,  # Can handle 2 concurrent jobs
    "unirig_root": "/path/to/UniRig"
}
```

## ⚙️ Configuration and Deployment

### Model Configuration File

Create `config/models.yaml`:

```yaml
adapters:
  trellis-text-to-mesh:
    class: "adapters.trellis_adapter.TrellisTextToMeshAdapter"
    config:
      vram_requirement: 6144
      max_concurrent: 1
      trellis_root: "/data/workspace/3DAIGC/Pocket3DStudio/backend/thirdparty/TRELLIS"
    
  hunyuan3d-text-to-mesh:
    class: "adapters.hunyuan3d_adapter.Hunyuan3DTextToMeshAdapter"
    config:
      vram_requirement: 8192
      max_concurrent: 1
      hunyuan3d_root: "/data/workspace/3DAIGC/Pocket3DStudio/backend/thirdparty/Hunyuan3D-2"

deployment:
  auto_load_adapters: true
  preferred_adapters:
    text_to_mesh: ["trellis-text-to-mesh"]
    image_to_mesh: ["hunyuan3d-image-to-mesh"] 
    auto_rig: ["unirig-auto-rigging"]
    mesh_segmentation: ["partfield-segmentation"]
```

### Production Deployment

```python
# In your main app startup
from config.adapter_registry import register_all_adapters
from core.scheduler import ModelScheduler, GPUMonitor

async def initialize_production_system():
    # Register all adapters
    register_all_adapters()
    
    # Create high-performance scheduler
    gpu_monitor = GPUMonitor(memory_buffer=1024)  # Keep 1GB free
    scheduler = ModelScheduler(
        gpu_monitor=gpu_monitor,
    )
    
    # Load production adapters
    production_adapters = [
        ("trellis-text-to-mesh", {"vram_requirement": 6144}),
        ("hunyuan3d-text-to-mesh", {"vram_requirement": 8192}),
        ("hunyuan3d-image-to-mesh", {"vram_requirement": 6144}),
        ("unirig-auto-rigging", {"vram_requirement": 4096}),
        ("partfield-segmentation", {"vram_requirement": 3072}),
    ]
    
    for adapter_name, config in production_adapters:
        adapter = adapter_registry.create_adapter_instance(adapter_name, **config)
        if adapter:
            scheduler.register_model(adapter)
    
    await scheduler.start()
    return scheduler
```

## 🧪 Testing and Debugging

### Test Your Adapter

```python
import asyncio
import pytest
from adapters.my_model_adapter import MyModelAdapter

@pytest.mark.asyncio
async def test_my_adapter():
    # Create adapter
    adapter = MyModelAdapter(
        vram_requirement=4096,
        max_concurrent=1
    )
    
    # Test loading
    await adapter.load(gpu_id=0)
    assert adapter.status == ModelStatus.LOADED
    
    # Test processing
    inputs = {
        "text_prompt": "Test prompt",
        "quality": "medium",
        "output_format": "glb"
    }
    
    result = await adapter.process(inputs)
    assert result["success"] == True
    assert "output_mesh_path" in result
    
    # Test unloading
    await adapter.unload()
    assert adapter.status == ModelStatus.UNLOADED
```

### Debug Common Issues

```python
# Enable debug logging
import logging
logging.getLogger("adapters").setLevel(logging.DEBUG)
logging.getLogger("core.models").setLevel(logging.DEBUG)

# Check VRAM usage
from core.scheduler import GPUMonitor
monitor = GPUMonitor()
gpu_info = await monitor.get_gpu_info()
print(f"Available VRAM: {gpu_info[0]['free_memory']}MB")

# Test adapter creation
try:
    adapter = adapter_registry.create_adapter_instance("my-adapter")
    print("✓ Adapter created successfully")
except Exception as e:
    print(f"✗ Adapter creation failed: {e}")
```

## 📚 Best Practices

### 1. Resource Management

```python
# Always implement proper cleanup
async def _unload_model(self):
    if self.model is not None:
        self.model.cpu()  # Move to CPU first
        del self.model    # Delete reference
        self.model = None
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 2. Error Handling

```python
async def _process_request(self, inputs):
    try:
        # Validate inputs first
        output_format = self._validate_common_inputs(inputs)
        
        # Process with timeouts for long operations
        result = await asyncio.wait_for(
            self._run_model(inputs),
            timeout=300  # 5 minute timeout
        )
        
        return self._create_response(result, output_format)
        
    except asyncio.TimeoutError:
        raise Exception("Model processing timed out")
    except ValueError as e:
        raise Exception(f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise Exception(f"Model processing failed: {str(e)}")
```

### 3. Performance Optimization

```python
class OptimizedAdapter(TextToMeshModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_cache = {}  # Cache repeated operations
        self._preprocessing_cache = {}
    
    async def _process_request(self, inputs):
        # Cache preprocessing results
        cache_key = self._get_cache_key(inputs)
        if cache_key in self._preprocessing_cache:
            preprocessed = self._preprocessing_cache[cache_key]
        else:
            preprocessed = self._preprocess(inputs)
            self._preprocessing_cache[cache_key] = preprocessed
        
        # Use mixed precision for speed
        with torch.cuda.amp.autocast():
            result = self.model(preprocessed)
        
        return self._postprocess(result, inputs)
```

### 4. Monitoring and Metrics

```python
import time
from core.utils.metrics import ModelMetrics

async def _process_request(self, inputs):
    start_time = time.time()
    
    try:
        result = await self._run_model(inputs)
        
        # Record success metrics
        processing_time = time.time() - start_time
        ModelMetrics.record_success(self.model_id, processing_time)
        
        return result
        
    except Exception as e:
        # Record failure metrics
        ModelMetrics.record_failure(self.model_id, str(e))
        raise
```

## 🎉 Summary

You now have a complete guide for implementing and using model adapters! The key points are:

1. **Inherit** from the appropriate base model class
2. **Implement** the three required methods: `_load_model()`, `_unload_model()`, `_process_request()`
3. **Register** your adapter with the registry
4. **Use** through the scheduler and API system

Your adapters will automatically get:
- ✅ VRAM-aware scheduling
- ✅ Concurrent processing limits  
- ✅ Load balancing across GPUs
- ✅ Standardized API endpoints
- ✅ Error handling and logging
- ✅ Automatic capacity management

Happy coding! 🚀
