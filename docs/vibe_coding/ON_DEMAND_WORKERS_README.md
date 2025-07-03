# On-Demand Worker Creation for MultiprocessModelScheduler

## Overview

The `MultiprocessModelScheduler` has been updated to create workers on-demand instead of creating all workers at startup. This provides better resource utilization and more flexible scaling based on actual demand.

## Key Changes

### 1. Model Configuration
- Added `max_workers` parameter to model registration
- Each model can now specify its maximum number of workers
- Default is 1 worker per model if not specified

```python
model_config = {
    "model_id": "text_to_mesh",
    "feature_type": "mesh_generation", 
    "vram_requirement": 2048,  # MB
    "max_workers": 3,  # Maximum workers for this model
    # ... other config
}
```

### 2. Worker Creation Logic

Workers are created only when:
1. **Demand exists**: All existing workers for a model are busy OR no workers exist for that model
2. **VRAM available**: Sufficient GPU memory is available (based on `vram_requirement`)
3. **Under limit**: The number of workers for the model hasn't reached `max_workers`

### 3. Worker Destruction Logic

Workers are destroyed when:
1. **Idle time**: Worker has been idle for more than `idle_cleanup_interval` (default: 30 seconds)
2. **VRAM needed**: When VRAM is needed for new workers, idle workers are destroyed to free memory
3. **Multiple workers**: At least one worker is kept per model when possible

### 4. VRAM Management

- Dynamic VRAM allocation based on model requirements
- Intelligent cleanup of idle workers when VRAM is constrained
- Longest-idle workers are destroyed first

## Benefits

1. **Resource Efficiency**: Only creates workers when needed
2. **Memory Optimization**: Automatically frees VRAM when not in use
3. **Scalability**: Scales workers up/down based on actual demand
4. **Model-Specific Limits**: Different models can have different worker limits

## Configuration Example

```python
# Register models with different worker limits
scheduler.register_model({
    "model_id": "lightweight_model",
    "feature_type": "mesh_generation",
    "vram_requirement": 1024,  # 1GB
    "max_workers": 5  # Can have up to 5 workers
})

scheduler.register_model({
    "model_id": "heavy_model", 
    "feature_type": "mesh_generation",
    "vram_requirement": 8192,  # 8GB
    "max_workers": 1  # Only 1 worker due to high VRAM usage
})
```

## Monitoring

Check system status to monitor worker creation/destruction:

```python
status = await scheduler.get_system_status()
print(f"Active workers: {status['scheduler']['num_workers']}")
print(f"Worker assignments: {scheduler.worker_assignments}")
```

## Testing

Run the test script to see on-demand worker creation in action:

```bash
python test_on_demand_scheduler.py
```

The test demonstrates:
- Initial startup with 0 workers
- Worker creation when jobs are scheduled
- Worker reuse for subsequent jobs
- Worker creation for different models/features
- Idle worker cleanup after timeout

## Implementation Details

### Key Methods Added/Modified:

1. **`_find_or_create_worker_for_job()`**: Main logic for finding existing or creating new workers
2. **`_create_worker_for_model()`**: Creates a new worker for a specific model
3. **`_ensure_vram_available()`**: Frees VRAM by destroying idle workers when needed
4. **`_cleanup_idle_workers()`**: Background task that cleans up long-idle workers
5. **`_destroy_worker()`**: Properly shuts down and cleans up a worker process

### Worker Lifecycle:

```
Job Request → Find Available Worker → (If none) Check Limits → Check VRAM → Create Worker → Process Job → Mark Idle → (After timeout) Destroy if excess
```

This implementation provides a much more efficient and scalable approach to worker management in the multiprocessing scheduler. 