#!/usr/bin/env python3
"""
Test script for on-demand worker creation in MultiprocessModelScheduler
"""

import asyncio
import logging
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.scheduler.gpu_monitor import GPUMonitor
from core.scheduler.job_queue import JobQueue, JobRequest
from core.scheduler.multiprocess_scheduler import MultiprocessModelScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_on_demand_workers():
    """Test on-demand worker creation"""

    # Create scheduler
    gpu_monitor = GPUMonitor()
    job_queue = JobQueue()
    scheduler = MultiprocessModelScheduler(gpu_monitor=gpu_monitor, job_queue=job_queue)

    # Register some test models with different max_workers
    test_models = [
        {
            "model_id": "trellis_text_to_textured_mesh",
            "feature_type": "text_to_textured_mesh",
            "module": "adapters.trellis_adapter",
            "class": "TrellisTextToTexturedMeshAdapter",
            "vram_requirement": 12800,  # 12.5GB
            "max_workers": 1,
        },
        {
            "model_id": "partfield_mesh_segmentation",
            "feature_type": "mesh_segmentation",
            "module": "adapters.partfield_adapter",
            "class": "PartFieldSegmentationAdapter",
            "vram_requirement": 3192,  # 1GB
            "max_workers": 1,
        },
    ]

    # Register models
    for model_config in test_models:
        scheduler.register_model(model_config)
        logger.info(f"Registered model: {model_config['model_id']}")

    # Start scheduler (no workers should be created yet)
    await scheduler.start()
    logger.info(f"Initial worker count: {len(scheduler.workers)}")

    try:
        # Test 1: Schedule job for mesh generation (should create first worker)
        logger.info("\n=== Test 1: First job for text_to_textured_mesh ===")
        job1 = JobRequest(
            feature="text_to_textured_mesh",
            inputs={"text_prompt": "A red car"},
            model_preference="trellis_text_to_textured_mesh",
        )

        job_id1 = await scheduler.schedule_job(job1)
        logger.info(f"Scheduled job {job_id1}")

        # Wait a bit for worker creation
        await asyncio.sleep(2)
        logger.info(f"Worker count after first job: {len(scheduler.workers)}")
        logger.info(f"Workers: {list(scheduler.workers.keys())}")

        # Test 2: Schedule another job for same model (should reuse worker if not busy)
        logger.info("\n=== Test 2: Second job for text_to_textured_mesh ===")
        job2 = JobRequest(
            feature="text_to_textured_mesh",
            inputs={"text_prompt": "A blue sphere"},
            model_preference="trellis_text_to_textured_mesh",
        )

        job_id2 = await scheduler.schedule_job(job2)
        logger.info(f"Scheduled job {job_id2}")

        # Wait a bit
        await asyncio.sleep(2)
        logger.info(f"Worker count after second job: {len(scheduler.workers)}")

        # Test 3: Schedule job for different feature (should create new worker)
        logger.info("\n=== Test 3: Job for segmentation ===")
        job3 = JobRequest(
            feature="mesh_segmentation",
            inputs={"mesh_path": "assets/example_mesh/typical_creature_furry.obj"},
            model_preference="partfield_mesh_segmentation",
        )

        job_id3 = await scheduler.schedule_job(job3)
        logger.info(f"Scheduled job {job_id3}")

        # Wait a bit
        await asyncio.sleep(2)
        logger.info(f"Worker count after segmentation job: {len(scheduler.workers)}")

        # Test 4: Check system status
        logger.info("\n=== Test 4: System Status ===")
        status = await scheduler.get_system_status()
        logger.info(f"System status: {status['scheduler']}")
        logger.info(f"Worker assignments: {scheduler.worker_assignments}")

        # Wait for jobs to potentially complete
        await asyncio.sleep(560)

        # Test 5: Check job statuses
        logger.info("\n=== Test 5: Job Status Check ===")
        for job_id in [job_id1, job_id2, job_id3]:
            status = await scheduler.get_job_status(job_id)
            if status:
                logger.info(f"Job {job_id}: {status['status']}")
            else:
                logger.info(f"Job {job_id}: Not found")

        # Wait for idle cleanup to potentially kick in
        logger.info("\n=== Test 6: Waiting for idle cleanup ===")
        await asyncio.sleep(40)  # Wait longer than idle_cleanup_interval
        logger.info(f"Worker count after idle cleanup: {len(scheduler.workers)}")

    finally:
        # Stop scheduler
        await scheduler.stop()
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    asyncio.run(test_on_demand_workers())
