"""
Scheduler Factory

This module provides a factory for creating multiprocessing schedulers
for AI model inference workloads.
"""

import logging
import multiprocessing as mp
from typing import Dict, List, Optional

from .gpu_monitor import GPUMonitor
from .job_queue import JobQueue
from .model_factory import (
    get_default_model_configs,
    get_model_configs_from_settings,
)
from .multiprocess_scheduler import MultiprocessModelScheduler

logger = logging.getLogger(__name__)


class SchedulerFactory:
    """Factory for creating multiprocessing schedulers"""

    @staticmethod
    def create_scheduler(
        gpu_monitor: Optional[GPUMonitor] = None,
        job_queue: Optional[JobQueue] = None,
        **kwargs,
    ) -> MultiprocessModelScheduler:
        """
        Create a multiprocessing scheduler.

        Args:
            gpu_monitor: Optional GPU monitor instance
            job_queue: Optional job queue instance
            **kwargs: Additional scheduler configuration

        Returns:
            MultiprocessModelScheduler instance
        """
        # Create shared components if not provided
        gpu_monitor = gpu_monitor or GPUMonitor()
        job_queue = job_queue or JobQueue()

        logger.info("Creating multiprocess scheduler (true parallelism)")

        # Set multiprocessing start method for CUDA safety
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
            logger.info("Set multiprocessing start method to 'spawn' for CUDA safety")

        return MultiprocessModelScheduler(gpu_monitor=gpu_monitor, job_queue=job_queue)

    @staticmethod
    def create_configured_scheduler(
        auto_register_models: bool = True,
        model_subset: Optional[List[str]] = None,
        reduce_vram_requirements: bool = False,
        models_config: Optional[Dict] = None,
        **kwargs,
    ) -> MultiprocessModelScheduler:
        """
        Create a pre-configured scheduler with models registered.

        Args:
            auto_register_models: Whether to automatically register known models
            model_subset: Optional list of specific model IDs to register
            reduce_vram_requirements: Whether to reduce VRAM requirements for testing
            models_config: Optional parsed models configuration from settings
            **kwargs: Additional scheduler configuration

        Returns:
            Configured scheduler instance with models registered
        """
        # Create scheduler
        scheduler = SchedulerFactory.create_scheduler(**kwargs)

        if auto_register_models:
            # Get model configurations
            if models_config is not None:
                all_model_configs = get_model_configs_from_settings(models_config)
            else:
                all_model_configs = get_default_model_configs()

            # Filter to subset if specified
            if model_subset:
                model_configs = {
                    model_id: config
                    for model_id, config in all_model_configs.items()
                    if model_id in model_subset
                }
            else:
                model_configs = all_model_configs

            # Register models
            logger.info(f"Registering {len(model_configs)} models with scheduler")
            for model_id, config in model_configs.items():
                # Optionally reduce VRAM requirements
                if reduce_vram_requirements:
                    config = config.copy()  # Don't modify original
                    config["vram_requirement"] = min(config["vram_requirement"], 4096)

                # Register with scheduler
                scheduler.register_model(config)
                logger.debug(f"Registered model: {model_id}")

        return scheduler


def create_production_scheduler(**kwargs) -> MultiprocessModelScheduler:
    """
    Create a production-ready scheduler with optimal configuration.

    This is the recommended way to create a scheduler for production use.

    Args:
        **kwargs: Override configuration parameters (including models_config)

    Returns:
        Configured scheduler instance
    """
    # Get optimal configuration
    optimal_config = {}
    optimal_config.update(kwargs)  # Allow overrides

    logger.info(f"Creating production scheduler with config: {optimal_config}")

    # Create and configure scheduler
    scheduler = SchedulerFactory.create_configured_scheduler(
        auto_register_models=True, **optimal_config
    )

    return scheduler


def create_development_scheduler(**kwargs) -> MultiprocessModelScheduler:
    """
    Create a development-friendly scheduler with reduced resource requirements.

    Args:
        **kwargs: Override configuration parameters (including models_config)

    Returns:
        Configured scheduler instance
    """
    # For development, prefer fewer resources and faster startup
    defaults = {
        "reduce_vram_requirements": True,
        "model_subset": [
            "trellis_text_to_textured_mesh",
            "partfield_mesh_segmentation",
        ],  # Subset of models for faster development
    }

    config = defaults.copy()
    config.update(kwargs)

    logger.info("Creating development scheduler with reduced requirements")

    scheduler = SchedulerFactory.create_configured_scheduler(
        auto_register_models=True, **config
    )

    return scheduler
