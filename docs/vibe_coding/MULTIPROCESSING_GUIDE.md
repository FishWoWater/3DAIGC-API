# Multiprocessing Inference Guide

This guide explains how to use the multiprocessing-based scheduler for true parallelism in 3D generative models inference.

## Overview

The multiprocessing scheduler (`MultiprocessModelScheduler`) provides true parallelism by spawning worker processes for each GPU, enabling efficient utilization of multiple GPUs and avoiding Python's Global Interpreter Lock (GIL) limitations.

### Key Benefits

- **True Parallelism**: Worker processes run independently, not limited by GIL
- **GPU Isolation**: Each worker process manages models on specific GPUs
- **Fault Tolerance**: Worker crashes don't affect the main process or other workers
- **Memory Management**: Better VRAM utilization and garbage collection
- **Scalability**: Easily scale to multiple GPUs and workers

## Quick Start

### Basic Usage

```python
import asyncio
from core.scheduler.scheduler_factory import create_production_scheduler

async def main():
    # Create a production-ready multiprocessing scheduler
    scheduler = create_production_scheduler()
    
    # Start the scheduler (spawns worker processes)
    await scheduler.start()
    
    try:
        # Your application logic here
        job_request = JobRequest(
            feature="text_to_textured_mesh",
            inputs={"text_prompt": "a cute robot toy"}
        )
        
        job_id = await scheduler.schedule_job(job_request)
        
        # Wait for completion
        while True:
            status = await scheduler.get_job_status(job_id)
            if status['status'] in ['completed', 'error']:
                break
            await asyncio.sleep(1)
        
        print(f"Job completed: {status}")
        
    finally:
        # Clean shutdown
        await scheduler.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Development Usage

```python
from core.scheduler.scheduler_factory import create_development_scheduler

# Create a development scheduler with reduced requirements
scheduler = create_development_scheduler(
    max_workers_per_gpu=1,
    model_subset=['trellis_text_to_textured_mesh'],
    reduce_vram_requirements=True
)
```

## Architecture

### Process Structure

```
Main Process (FastAPI Server)
├── GPU Monitor (thread)
├── Job Queue Manager (asyncio)
└── Worker Processes
    ├── Worker_GPU0_0 (Process)
    │   ├── Model A (TRELLIS)
    │   └── Model B (HunyuanAD) 
    ├── Worker_GPU0_1 (Process)
    │   ├── Model C (PartField)
    │   └── Model D (UniRig)
    ├── Worker_GPU1_0 (Process)
    │   └── ...
    └── ...
```

### Communication Flow

1. **Job Submission**: API receives request → Job Queue
2. **Job Assignment**: Scheduler finds suitable worker → Sends job to worker queue
3. **Processing**: Worker loads model → Processes job → Returns result
4. **Result Handling**: Main process receives result → Updates job status

## Configuration

### Scheduler Types

```python
from core.scheduler.scheduler_factory import SchedulerFactory, SchedulerType

# Multiprocessing scheduler (recommended)
scheduler = SchedulerFactory.create_scheduler(
    scheduler_type=SchedulerType.MULTIPROCESS,
    max_workers_per_gpu=2,  # Number of workers per GPU
    max_models_per_worker=2  # Models each worker can handle
)

# Async scheduler (for comparison)
scheduler = SchedulerFactory.create_scheduler(
    scheduler_type=SchedulerType.ASYNC,
)
```

### Worker Configuration

```python
# Conservative setup (safer for limited VRAM)
scheduler = MultiprocessModelScheduler(
    max_workers_per_gpu=1,    # 1 worker per GPU
    max_models_per_worker=1   # 1 model per worker
)

# Aggressive setup (requires more VRAM)
scheduler = MultiprocessModelScheduler(
    max_workers_per_gpu=3,    # 3 workers per GPU
    max_models_per_worker=2   # 2 models per worker
)
```

### Model Registration

```python
# Manual registration
from core.scheduler.model_factory import ModelFactory

config = ModelFactory.create_model_config(
    model_id="custom_model",
    feature_type="text_to_mesh",
    vram_requirement=4096,
    init_params={"custom_param": "value"}
)

scheduler.register_model(config)

# Automatic registration
from core.scheduler.model_factory import get_model_configs_for_adapters

configs = get_model_configs_for_adapters()
for model_id, config in configs.items():
    scheduler.register_model(config)
```

## Best Practices

### 1. GPU Memory Management

```python
# Monitor GPU memory usage
status = await scheduler.get_system_status()
for gpu in status['gpu']:
    memory_usage = gpu['memory_utilization']
    if memory_usage > 0.9:
        print(f"GPU {gpu['id']} memory high: {memory_usage:.1%}")
```

### 2. Error Handling

```python
try:
    job_id = await scheduler.schedule_job(job_request)
except Exception as e:
    if "No models available" in str(e):
        # Handle missing model case
        print("Model not available, try different feature type")
    else:
        # Handle other errors
        print(f"Scheduling failed: {e}")
```

### 3. Graceful Shutdown

```python
import signal
import asyncio

async def shutdown_handler(scheduler):
    """Graceful shutdown handler"""
    print("Shutting down scheduler...")
    await scheduler.stop()
    print("Scheduler stopped")

# Register signal handlers
def signal_handler(signum, frame):
    asyncio.create_task(shutdown_handler(scheduler))

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

### 4. Performance Monitoring

```python
import time

async def monitor_performance(scheduler):
    """Monitor scheduler performance"""
    while True:
        status = await scheduler.get_system_status()
        
        # Log queue status
        queue_info = status['queue']
        print(f"Queue: {queue_info.get('queued', 0)} queued, "
              f"{queue_info.get('processing', 0)} processing")
        
        # Log worker status
        for worker_id, worker_info in status['workers'].items():
            if 'error' not in worker_info:
                print(f"{worker_id}: {len(worker_info['loaded_models'])} models, "
                      f"{worker_info['processing_jobs']} jobs")
        
        await asyncio.sleep(10)
```

## Troubleshooting

### Common Issues

1. **"No GPU has enough memory for this model"**
   - Reduce `max_workers_per_gpu`
   - Reduce `max_models_per_worker`
   - Use smaller models or reduce VRAM requirements

2. **Worker processes crashing**
   - Check CUDA compatibility
   - Verify model files exist
   - Check for memory leaks in model code

3. **Slow job processing**
   - Increase `max_workers_per_gpu` if VRAM allows
   - Check GPU utilization
   - Profile model loading/inference times

4. **Import errors in workers**
   - Ensure all dependencies are installed
   - Check Python path in worker processes
   - Verify model adapter imports

### Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Monitor worker health
async def check_worker_health(scheduler):
    status = await scheduler.get_system_status()
    for worker_id, worker_info in status['workers'].items():
        if 'error' in worker_info:
            print(f"Worker {worker_id} error: {worker_info['error']}")
        else:
            print(f"Worker {worker_id} healthy")
```

### Performance Tuning

1. **VRAM Optimization**
   ```python
   # Start with conservative settings
   max_workers_per_gpu = 1
   max_models_per_worker = 1
   
   # Gradually increase based on monitoring
   if average_vram_usage < 0.7:
       max_workers_per_gpu = 2
   ```

2. **CPU Utilization**
   ```python
   import multiprocessing as mp
   
   # Scale workers based on CPU count
   cpu_count = mp.cpu_count()
   max_workers_per_gpu = min(2, cpu_count // gpu_count)
   ```

3. **Load Balancing**
   ```python
   # Distribute models across workers by VRAM requirement
   heavy_models = ['trellis_text_to_textured_mesh']
   light_models = ['partfield_mesh_segmentation']
   
   # Assign heavy models to dedicated workers
   # Assign multiple light models to shared workers
   ```

## Integration with FastAPI

### API Dependencies

```python
from fastapi import Depends
from core.scheduler.scheduler_factory import create_production_scheduler

# Global scheduler instance
_scheduler = None

async def get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = create_production_scheduler()
        await _scheduler.start()
    return _scheduler

# Use in API endpoints
@app.post("/generate-mesh")
async def generate_mesh(
    request: MeshRequest,
    scheduler = Depends(get_scheduler)
):
    job_id = await scheduler.schedule_job(job_request)
    return {"job_id": job_id}
```

### Startup/Shutdown Events

```python
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global scheduler
    scheduler = create_production_scheduler()
    await scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    if scheduler:
        await scheduler.stop()
```

## Comparison: Async vs Multiprocessing

| Aspect | Async Scheduler | Multiprocessing Scheduler |
|--------|----------------|---------------------------|
| Parallelism | Concurrency only (GIL limited) | True parallelism |
| GPU Utilization | Sequential model loading | Parallel model loading |
| Memory Usage | Shared memory space | Isolated memory per worker |
| Fault Tolerance | Single point of failure | Worker isolation |
| Startup Time | Fast | Slower (process spawning) |
| Resource Usage | Lower overhead | Higher overhead |
| Scaling | Limited by GIL | Scales with hardware |

## Conclusion

The multiprocessing scheduler is recommended for production deployments where:
- Multiple GPUs are available
- High throughput is required
- Multiple concurrent users need service
- Fault tolerance is important

Use the async scheduler for:
- Development and testing
- Single GPU setups
- Lightweight deployments
- Situations where startup time is critical 