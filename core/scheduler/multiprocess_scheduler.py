"""
IMPORTANT: FIFO Job Processing Implementation

This multiprocessing scheduler has been modified to implement strict FIFO (First-In-First-Out)
job processing with the following key behavioral changes:

1. **No Retry Mechanism for Resource Constraints**: When a job cannot be processed due to
   WORKERS_BUSY or NO_VRAM, it is immediately put back at the front of the queue instead
   of using a retry mechanism with exponential backoff delays.

2. **Strict FIFO Order**: Jobs are processed in the exact order they were submitted.
   A job submitted later will NEVER be processed before an earlier job unless the
   earlier job is impossible to process.

3. **Job Failure Only for Impossible Cases**: Jobs are only failed if they are truly
   impossible to process:
   - No models available for the requested feature
   - VRAM requirement exceeds total available VRAM across all GPUs
   - Job has been waiting for more than 1 hour

4. **Resource-Based Requeuing**: When resources are unavailable, jobs are put back at
   the front of the queue and processing continues with a short delay to prevent
   busy waiting.

5. **Transient Error Handling**: Transient errors (network, timeout, etc.) cause jobs
   to be requeued rather than failed, maintaining the FIFO order.

6. **Sequential Processing**: Jobs are processed one at a time (not concurrently) to
   maintain strict ordering. Each job must complete before the next one starts.

7. **Single Instance**: Uses singleton pattern to prevent multiple scheduler instances
   when deployed with uvicorn. For true multi-worker deployments, use external queue systems.

This ensures predictable, ordered job processing while efficiently managing GPU resources.

DEPLOYMENT NOTE:
- Use `uvicorn app:main --workers 1` for single-worker deployment with this scheduler
- For multi-worker deployments, consider Redis-based job queues or similar external systems
"""

import asyncio
import logging
import multiprocessing as mp
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..models.base import BaseModel
from .gpu_monitor import GPUMonitor
from .job_queue import JobQueue, JobRequest

logger = logging.getLogger(__name__)

# Whether to retry transient errors (timeout, connection, network, temporarily)
RETRY_TRANSIENT_ERRORS = False


class WorkerConfig:
    """Configuration for a worker process - simplified to one model per worker"""

    def __init__(self, worker_id: str, gpu_id: int, model_config: Dict[str, Any]):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.model_config = model_config  # Single model configuration for this worker


class WorkerMessage:
    """Message types for worker communication"""

    class Type:
        PROCESS_JOB = "process_job"
        LOAD_MODEL = "load_model"
        UNLOAD_MODEL = "unload_model"
        GET_STATUS = "get_status"
        SHUTDOWN = "shutdown"
        HEALTH_CHECK = "health_check"

    def __init__(self, msg_type: str, data: Any = None, msg_id: Optional[str] = None):
        self.type = msg_type
        self.data = data
        self.msg_id = msg_id or str(uuid.uuid4())
        self.timestamp = time.time()


class WorkerResponse:
    """Response from worker process"""

    def __init__(
        self, msg_id: str, success: bool, data: Any = None, error: Optional[str] = None
    ):
        self.msg_id = msg_id
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = time.time()


def model_worker_process(
    worker_config: WorkerConfig,
    job_queue: mp.Queue,
    response_queue: mp.Queue,
    control_queue: mp.Queue,
    control_response_queue: mp.Queue,
):
    """
    Main worker process function that handles model loading and job processing.

    This function runs in a separate process and manages models on a specific GPU.
    """
    worker_id = worker_config.worker_id
    gpu_id = worker_config.gpu_id

    try:
        # Set up process environment
        logger.info(f"Starting worker process {worker_id} on GPU {gpu_id}")

        # Set CUDA device first thing
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            # Warm up CUDA context
            dummy = torch.zeros(1, device=f"cuda:{gpu_id}")
            del dummy
            torch.cuda.empty_cache()

        # Initialize worker state - simplified for single model
        loaded_model: Optional[BaseModel] = None
        model_config = worker_config.model_config
        model_id = model_config["model_id"]
        processing_job: Optional[str] = None

        logger.info(f"Worker {worker_id} initialized for model {model_id}")

        # Load model immediately upon worker creation
        try:
            model = _create_model_from_config(model_config)
            success = model.load(gpu_id)

            if success:
                loaded_model = model
                logger.info(f"Worker {worker_id} successfully loaded model {model_id}")
            else:
                logger.error(f"Worker {worker_id} failed to load model {model_id}")
                return  # Exit worker if model loading fails
        except Exception as e:
            logger.error(f"Worker {worker_id} failed to load model {model_id}: {e}")
            return  # Exit worker if model loading fails

        # Main worker loop
        while True:
            try:
                # Check for control messages (non-blocking)
                try:
                    control_msg = control_queue.get_nowait()
                    response = _handle_control_message(
                        control_msg,
                        loaded_model,
                        model_config,
                        processing_job,
                        gpu_id,
                    )
                    control_response_queue.put(response)

                    # Check for shutdown
                    if control_msg.type == WorkerMessage.Type.SHUTDOWN:
                        break

                except queue.Empty:
                    pass

                # Check for job processing (blocking with timeout)
                try:
                    job_data = job_queue.get(timeout=0.1)
                    job_request, result_callback_id = job_data
                    logger.info(f"Worker {worker_id} received job {job_request.job_id}")

                    # Process job
                    result, loaded_model, processing_job = _process_job_in_worker(
                        job_request,
                        loaded_model,
                        model_config,
                        processing_job,
                        gpu_id,
                    )
                    logger.info(
                        f"Worker {worker_id} processed job {job_request.job_id}, result: {result}"
                    )

                    # Send result back
                    response_queue.put((result_callback_id, result))
                    logger.info(
                        f"Worker {worker_id} sent result for job {job_request.job_id}"
                    )

                except queue.Empty:
                    continue

            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id} received shutdown signal")
    except Exception as e:
        logger.error(f"Fatal error in worker {worker_id}: {e}")
    finally:
        # Cleanup
        try:
            if loaded_model and hasattr(loaded_model, "_unload_model"):
                # Synchronous unload for cleanup
                try:
                    loaded_model._unload_model()
                except:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        logger.info(f"Worker {worker_id} shutdown complete")


def _handle_control_message(
    msg: WorkerMessage,
    loaded_model: Optional[BaseModel],
    model_config: Dict[str, Any],
    processing_job: Optional[str],
    gpu_id: int,
) -> WorkerResponse:
    """Handle control messages in worker process - simplified for single model"""
    try:
        model_id = model_config["model_id"]

        if msg.type == WorkerMessage.Type.LOAD_MODEL:
            requested_model_id = msg.data["model_id"]
            if requested_model_id != model_id:
                return WorkerResponse(
                    msg.msg_id,
                    False,
                    error=f"Worker only supports model {model_id}, not {requested_model_id}",
                )

            if loaded_model is not None:
                return WorkerResponse(msg.msg_id, True, {"status": "already_loaded"})

            # Create and load model
            model = _create_model_from_config(model_config)

            # Load synchronously (we're in a worker process)
            success = model.load(gpu_id)

            if success:
                loaded_model = model
                return WorkerResponse(msg.msg_id, True, {"status": "loaded"})
            else:
                return WorkerResponse(msg.msg_id, False, error="Failed to load model")

        elif msg.type == WorkerMessage.Type.UNLOAD_MODEL:
            requested_model_id = msg.data["model_id"]
            if requested_model_id != model_id:
                return WorkerResponse(msg.msg_id, True, {"status": "not_our_model"})

            if loaded_model is None:
                return WorkerResponse(msg.msg_id, True, {"status": "not_loaded"})

            success = loaded_model.unload()

            if success:
                loaded_model = None
                return WorkerResponse(msg.msg_id, True, {"status": "unloaded"})
            else:
                return WorkerResponse(msg.msg_id, False, error="Failed to unload model")

        elif msg.type == WorkerMessage.Type.GET_STATUS:
            status = {
                "gpu_id": gpu_id,
                "model_id": model_id,
                "loaded": loaded_model is not None,
                "processing": processing_job is not None,
                "model_status": loaded_model.get_info() if loaded_model else None,
            }
            return WorkerResponse(msg.msg_id, True, status)

        elif msg.type == WorkerMessage.Type.HEALTH_CHECK:
            return WorkerResponse(msg.msg_id, True, {"status": "healthy"})

        elif msg.type == WorkerMessage.Type.SHUTDOWN:
            return WorkerResponse(msg.msg_id, True, {"status": "shutting_down"})

        else:
            return WorkerResponse(
                msg.msg_id, False, error=f"Unknown message type: {msg.type}"
            )

    except Exception as e:
        logger.error(f"Error handling control message {msg.type}: {e}")
        return WorkerResponse(msg.msg_id, False, error=str(e))


def _process_job_in_worker(
    job_request: JobRequest,
    loaded_model: Optional[BaseModel],
    model_config: Dict[str, Any],
    processing_job: Optional[str],
    gpu_id: int,
) -> Tuple[Dict[str, Any], Optional[BaseModel], Optional[str]]:
    """Process a job in the worker process - simplified for single model"""
    try:
        job_id = job_request.job_id
        model_id = model_config["model_id"]

        # Check if this worker can handle the requested feature
        if model_config.get("feature_type") != job_request.feature:
            return (
                {
                    "success": False,
                    "error": f"Worker model {model_id} handles {model_config.get('feature_type')}, not {job_request.feature}",
                    "job_id": job_id,
                },
                loaded_model,
                processing_job,
            )

        # Check if worker is busy
        if processing_job is not None:
            return (
                {
                    "success": False,
                    "error": f"Worker is busy processing job {processing_job}",
                    "job_id": job_id,
                },
                loaded_model,
                processing_job,
            )

        # Load model if not already loaded
        if loaded_model is None:
            model = _create_model_from_config(model_config)
            success = model.load(gpu_id)

            if not success:
                return (
                    {
                        "success": False,
                        "error": f"Failed to load model {model_id}",
                        "job_id": job_id,
                    },
                    loaded_model,
                    processing_job,
                )

            loaded_model = model

        # Mark as processing
        processing_job = job_id

        # Process job
        result = loaded_model._process_request(job_request.inputs)

        return (
            {
                "success": True,
                "result": result,
                "job_id": job_id,
                "model_id": model_id,
            },
            loaded_model,
            None,
        )  # Clear processing_job

    except Exception as e:
        logger.error(f"Error processing job {job_request.job_id}: {e}")
        return (
            {"success": False, "error": str(e), "job_id": job_request.job_id},
            loaded_model,
            None,
        )  # Clear processing_job on error


def _create_model_from_config(config: Dict[str, Any]) -> BaseModel:
    """Create a model instance from configuration"""
    from .model_factory import create_model_from_config

    return create_model_from_config(config)


class MultiprocessModelScheduler:
    """
    Multiprocessing-based model scheduler that provides true parallelism with strict FIFO processing.

    This scheduler spawns worker processes on-demand for each model, enabling true parallelism
    for AI inference workloads while maintaining strict FIFO job processing order.

    IMPORTANT: This scheduler should be used with uvicorn --workers 1 to avoid
    multiple scheduler instances. For multi-worker deployments, use external
    job queue systems like Redis/Celery.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MultiprocessModelScheduler, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        gpu_monitor: Optional[GPUMonitor] = None,
        job_queue: Optional[JobQueue] = None,
    ):
        # Prevent multiple initialization
        if self._initialized:
            return

        # Set multiprocessing start method for CUDA safety
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)

        self.gpu_monitor = gpu_monitor or GPUMonitor(tracking_mode=True)
        self.job_queue = job_queue or JobQueue(
            persistence_file="data/job_queue_state.json", persistence_interval=30
        )

        # Worker management
        self.workers: Dict[str, mp.Process] = {}
        self.worker_configs: Dict[str, WorkerConfig] = {}
        self.worker_queues: Dict[str, mp.Queue] = {}
        self.worker_response_queues: Dict[str, mp.Queue] = {}
        self.worker_control_queues: Dict[str, mp.Queue] = {}
        self.worker_control_response_queues: Dict[str, mp.Queue] = {}

        # Model management
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.model_features: Dict[str, List[str]] = {}
        self.worker_assignments: Dict[
            str, List[str]
        ] = {}  # model_id -> list of worker_ids
        self.worker_status: Dict[str, bool] = {}  # worker_id -> is_busy
        self.model_max_workers: Dict[str, int] = {}  # model_id -> max_workers
        self.worker_last_used: Dict[
            str, float
        ] = {}  # worker_id -> timestamp of last use
        self.idle_cleanup_interval: float = 30.0  # seconds to wait before considering cleanup, about 20 mins (30 seconds for debug)

        # Scheduler state
        self.running = False
        self.response_handler_thread: Optional[threading.Thread] = None
        self.pending_results: Dict[str, asyncio.Future] = {}
        self.result_lock = threading.Lock()
        self.main_event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Thread pool for async operations
        self.thread_executor = ThreadPoolExecutor(max_workers=4)

        # Job processing task
        self.job_processing_task: Optional[asyncio.Task] = None

        # Mark as initialized
        self._initialized = True

    def register_model(self, model_config: Dict[str, Any]):
        """
        Register a model configuration with the scheduler.

        Args:
            model_config: Dictionary containing model configuration:
                - model_id: Unique model identifier
                - feature_type: Type of feature this model handles
                - module: Python module path for the model class
                - class: Model class name
                - init_params: Parameters for model initialization
                - vram_requirement: VRAM requirement in MB
                - max_workers: Maximum number of workers for this model (default: 1)
        """
        model_id = model_config["model_id"]
        feature_type = model_config["feature_type"]
        max_workers = model_config.get("max_workers", 1)

        self.model_registry[model_id] = model_config
        self.model_max_workers[model_id] = max_workers

        # Index by feature type
        if feature_type not in self.model_features:
            self.model_features[feature_type] = []
        self.model_features[feature_type].append(model_id)

        logger.info(
            f"Registered model: {model_id} for feature: {feature_type}, max_workers: {max_workers}"
        )

    def unregister_model(self, model_id: str):
        """Unregister a model from the scheduler"""
        if model_id in self.model_registry:
            config = self.model_registry[model_id]
            feature_type = config["feature_type"]

            # Remove from feature index
            if feature_type in self.model_features:
                if model_id in self.model_features[feature_type]:
                    self.model_features[feature_type].remove(model_id)
                if not self.model_features[feature_type]:
                    del self.model_features[feature_type]

            # Shutdown and remove workers for this model
            if model_id in self.worker_assignments:
                worker_ids = self.worker_assignments[model_id].copy()
                for worker_id in worker_ids:
                    asyncio.create_task(self._destroy_worker(worker_id))
                del self.worker_assignments[model_id]

            # Cleanup tracking
            if model_id in self.model_max_workers:
                del self.model_max_workers[model_id]

            del self.model_registry[model_id]
            logger.info(f"Unregistered model: {model_id}")

    async def schedule_job(self, job_request: JobRequest) -> str:
        """Schedule a new job for processing"""
        # Validate that we have models for this feature
        if job_request.feature not in self.model_features:
            raise Exception(
                f"No models available for feature: {job_request.feature}. Available features: {list(self.model_features.keys())}"
            )

        job_id = await self.job_queue.enqueue(job_request)
        logger.info(f"Scheduled job {job_id} for feature {job_request.feature}")

        # Job will be processed by the background job processing loop
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status and results"""
        job = await self.job_queue.get_job(job_id)
        if job:
            return job.to_dict()
        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if possible"""
        return await self.job_queue.cancel_job(job_id)

    async def start(self):
        """Start the scheduler - no workers created initially"""
        if self.running:
            logger.warning("Scheduler is already running")
            return

        self.running = True
        # Store reference to main event loop
        self.main_event_loop = asyncio.get_event_loop()
        logger.info("Starting multiprocess model scheduler (on-demand mode)")

        # Start job queue persistence
        await self.job_queue.start_persistence()

        # Start response handler thread
        self.response_handler_thread = threading.Thread(
            target=self._handle_worker_responses, daemon=True
        )
        self.response_handler_thread.start()

        # Start background job processing task
        self.job_processing_task = asyncio.create_task(self._job_processing_loop())

        # Start cleanup task for idle workers
        asyncio.create_task(self._cleanup_idle_workers())

        # Start job queue cleanup task
        asyncio.create_task(self._job_queue_cleanup_loop())

        logger.info("Scheduler started - workers will be created on demand")

    async def stop(self):
        """Stop the scheduler and cleanup worker processes"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping multiprocess model scheduler")

        # Stop job processing task
        if self.job_processing_task:
            self.job_processing_task.cancel()
            try:
                await self.job_processing_task
            except asyncio.CancelledError:
                pass

        # Stop job queue persistence
        await self.job_queue.stop_persistence()

        # Send shutdown messages to all workers
        for worker_id in list(self.workers.keys()):
            try:
                await self._destroy_worker(worker_id)
            except Exception as e:
                logger.warning(f"Error shutting down worker {worker_id}: {e}")

        # Stop response handler thread
        if self.response_handler_thread:
            self.response_handler_thread.join(timeout=2.0)

        # Final cleanup
        self.workers.clear()
        self.worker_configs.clear()
        self.worker_queues.clear()
        self.worker_response_queues.clear()
        self.worker_assignments.clear()
        self.worker_status.clear()
        self.worker_last_used.clear()
        self.main_event_loop = None

        logger.info("Multiprocess scheduler stopped")

    def validate_model_preference(self, model_id: str, feature: str) -> bool:
        """
        Validate that a model preference is valid for the given feature.

        Args:
            model_id: The preferred model ID
            feature: The feature type for the job

        Returns:
            True if the model exists and supports the feature, False otherwise
        """
        if not model_id:
            return True  # No preference is always valid

        # Check if model exists in registry
        if model_id not in self.model_registry:
            return False

        # Check if model supports the requested feature
        if feature not in self.model_features:
            return False

        return model_id in self.model_features[feature]

    def get_available_models(
        self, feature: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Get available models, optionally filtered by feature.

        Args:
            feature: Optional feature type to filter by

        Returns:
            Dictionary mapping feature types to lists of model IDs
        """
        if feature:
            return {feature: self.model_features.get(feature, [])}
        return dict(self.model_features)

    def _is_job_impossible(self, job_request: JobRequest) -> tuple[bool, str]:
        """
        Check if a job is impossible to process.

        Args:
            job_request: The job to check

        Returns:
            Tuple of (is_impossible, reason)
        """
        feature = job_request.feature

        # Check if any models support this feature
        if feature not in self.model_features:
            return True, f"No models available for feature: {feature}"

        # Check if job has been waiting too long (1 hour)
        if job_request.is_waiting_too_long():
            return True, "Job has been waiting for more than 1 hour"

        # Find the model that would be used for this job
        target_model_id = None
        if job_request.model_preference:
            if (
                job_request.model_preference in self.model_registry
                and job_request.model_preference in self.model_features.get(feature, [])
            ):
                target_model_id = job_request.model_preference

        if not target_model_id:
            available_models = self.model_features.get(feature, [])
            if not available_models:
                return True, f"No models available for feature: {feature}"
            target_model_id = available_models[0]

        # Check if VRAM requirement exceeds total available VRAM across all GPUs
        model_config = self.model_registry[target_model_id]
        vram_requirement = model_config.get("vram_requirement", 1024)

        gpu_status = self.gpu_monitor.get_gpu_status()
        max_gpu_vram = max((gpu["memory_total"] for gpu in gpu_status), default=0)

        if vram_requirement > max_gpu_vram:
            return (
                True,
                f"VRAM requirement ({vram_requirement}MB) exceeds maximum GPU VRAM ({max_gpu_vram}MB)",
            )

        return False, ""

    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        gpu_status = self.gpu_monitor.get_gpu_status()
        queue_status = await self.job_queue.get_queue_status()

        # Get worker status
        worker_status = {}
        for worker_id in self.workers:
            try:
                status_response = await self._send_control_message(
                    worker_id, WorkerMessage.Type.GET_STATUS, timeout=2.0
                )
                worker_status[worker_id] = (
                    status_response.data
                    if status_response.success
                    else {"error": status_response.error}
                )
            except Exception as e:
                worker_status[worker_id] = {"error": str(e)}

        return {
            "scheduler": {
                "type": "multiprocess_fifo",
                "running": self.running,
                "num_workers": len(self.workers),
                "processing_mode": "strict_fifo",
            },
            "gpu": gpu_status,
            "queue": queue_status,
            "workers": worker_status,
            "features": {
                feature: len(models) for feature, models in self.model_features.items()
            },
        }

    async def get_queue_info(self) -> Dict:
        """Get detailed queue information for monitoring FIFO behavior"""
        queue_status = await self.job_queue.get_queue_status()

        # Get the first few jobs in queue to verify FIFO order
        queue_preview = []
        async with self.job_queue._lock:
            for i, job in enumerate(list(self.job_queue._queue)[:5]):  # First 5 jobs
                queue_preview.append(
                    {
                        "position": i + 1,
                        "job_id": job.job_id,
                        "feature": job.feature,
                        "created_at": job.created_at.isoformat(),
                        "priority": job.priority,
                        "waiting_time_seconds": (
                            datetime.utcnow() - job.created_at
                        ).total_seconds(),
                    }
                )

        return {
            "queue_status": queue_status,
            "queue_preview": queue_preview,
            "fifo_processing": True,
            "retry_mechanism": "disabled_for_resources",
        }

    async def _job_processing_loop(self):
        """Background loop that processes jobs from the queue using JobQueue.dequeue()"""
        logger.info("Starting job processing loop")

        while self.running:
            try:
                # Try to get next job from queue
                job_request = await self.job_queue.dequeue()

                if job_request is None:
                    # No jobs available, wait a bit
                    await asyncio.sleep(0.5)
                    continue

                logger.info(f"Dequeued job {job_request.job_id} for processing")

                # Process job synchronously to maintain strict FIFO order
                await self._process_job_with_queue_integration(job_request)

            except Exception as e:
                logger.error(f"Error in job processing loop: {e}")
                await asyncio.sleep(1.0)

        logger.info("Job processing loop stopped")

    async def _process_job_with_queue_integration(self, job_request: JobRequest):
        """Process a job with proper JobQueue integration - FIFO approach"""
        try:
            # First check if job is impossible to process
            is_impossible, impossible_reason = self._is_job_impossible(job_request)
            if is_impossible:
                await self.job_queue.fail_job(job_request.job_id, impossible_reason)
                logger.error(
                    f"Job {job_request.job_id} is impossible: {impossible_reason}"
                )
                return

            # Try to find or create worker with suitable model
            worker_result = await self._find_or_create_worker_for_job(job_request)

            if worker_result == "NO_FEATURE":
                # No models available for this feature - fail immediately
                await self.job_queue.fail_job(
                    job_request.job_id,
                    f"No models available for feature: {job_request.feature}",
                )
                return
            elif worker_result in ["WORKERS_BUSY", "NO_VRAM"]:
                # Resources unavailable - put job back at front of queue and wait
                logger.info(
                    f"Job {job_request.job_id} cannot be processed now ({worker_result}), putting back in queue"
                )
                await self.job_queue.requeue_job(job_request.job_id)

                # Add a short delay to prevent busy waiting
                await asyncio.sleep(2.0)
                return
            elif isinstance(worker_result, str):
                # Got a worker ID - proceed with job processing
                worker_id = worker_result
            else:
                # Unexpected result
                await self.job_queue.fail_job(
                    job_request.job_id,
                    f"Unexpected error finding worker: {worker_result}",
                )
                return

            # Determine model ID for this job
            model_id = self._get_model_id_for_job(job_request)

            # Mark job as started in the queue
            await self.job_queue.mark_job_started(job_request.job_id, model_id)

            # Create future for result
            result_future = asyncio.Future()
            callback_id = str(uuid.uuid4())

            with self.result_lock:
                self.pending_results[callback_id] = result_future

            # Mark worker as busy and send job
            self._mark_worker_busy(worker_id)
            job_data = (job_request, callback_id)
            self.worker_queues[worker_id].put(job_data)
            logger.info(f"Sent job {job_request.job_id} to worker {worker_id}")

            # Wait for result
            logger.info(f"Waiting for result for job {job_request.job_id}")
            result = await result_future
            logger.info(f"Received result for job {job_request.job_id}: {result}")

            # Update job status through JobQueue
            if result["success"]:
                await self.job_queue.complete_job(job_request.job_id, result["result"])
                logger.info(f"Job {job_request.job_id} completed successfully")
            else:
                await self.job_queue.fail_job(job_request.job_id, result["error"])
                logger.info(f"Job {job_request.job_id} failed: {result['error']}")

        except Exception as e:
            logger.error(f"Error processing job {job_request.job_id}: {e}")

            # For transient errors, requeue the job; for non-transient errors, fail immediately
            error_msg = str(e).lower()
            if RETRY_TRANSIENT_ERRORS and any(
                keyword in error_msg
                for keyword in ["timeout", "connection", "network", "temporarily"]
            ):
                # Transient error - put job back in queue
                logger.info(
                    f"Job {job_request.job_id} encountered transient error, requeuing: {str(e)}"
                )
                await self.job_queue.requeue_job(job_request.job_id)
                await asyncio.sleep(2.0)  # Brief delay before processing continues
            else:
                # Non-transient error - fail immediately
                await self.job_queue.fail_job(job_request.job_id, str(e))

    def _get_model_id_for_job(self, job_request: JobRequest) -> str:
        """Get the model ID that will be used for processing this job"""
        feature = job_request.feature

        # Determine target model
        if job_request.model_preference:
            # Check if the preferred model exists and supports this feature
            if (
                job_request.model_preference in self.model_registry
                and job_request.model_preference in self.model_features.get(feature, [])
            ):
                return job_request.model_preference

        # If no preference or preference unavailable, choose first available model for feature
        available_models = self.model_features.get(feature, [])
        if available_models:
            return available_models[0]  # Simple selection, could be improved

        # This shouldn't happen if _is_job_impossible was called first, but just in case
        logger.warning(
            f"No model available for feature {feature} in job {job_request.job_id}"
        )
        return "unknown"

    async def _find_or_create_worker_for_job(self, job_request: JobRequest) -> str:
        """Find an existing worker or create a new one for the job

        Returns:
            - worker_id (str): If a worker is available or created successfully
            - "NO_FEATURE": If no models support this feature
            - "WORKERS_BUSY": If all workers are busy and max workers reached
            - "NO_VRAM": If no VRAM is available for creating new workers
        """
        feature = job_request.feature

        if feature not in self.model_features:
            return "NO_FEATURE"

        # Determine target model
        target_model_id = None
        if job_request.model_preference:
            # Check if the preferred model exists and supports this feature
            if (
                job_request.model_preference in self.model_registry
                and job_request.model_preference in self.model_features.get(feature, [])
            ):
                target_model_id = job_request.model_preference
            else:
                logger.warning(
                    f"Preferred model {job_request.model_preference} not available for feature {feature}"
                )

        # If no preference or preference unavailable, choose first available model for feature
        if not target_model_id:
            available_models = self.model_features.get(feature, [])
            if not available_models:
                return "NO_FEATURE"
            target_model_id = available_models[0]  # Simple selection, could be improved

        # Try to find existing available worker for this model
        if target_model_id in self.worker_assignments:
            worker_ids = self.worker_assignments[target_model_id]
            available_worker = self._find_available_worker(worker_ids)
            if available_worker:
                logger.info(
                    f"Using existing worker {available_worker} for model {target_model_id}"
                )
                return available_worker

        # Check if we can create a new worker for this model
        current_workers = len(self.worker_assignments.get(target_model_id, []))
        max_workers = self.model_max_workers.get(target_model_id, 1)

        if current_workers >= max_workers:
            logger.info(
                f"Model {target_model_id} has reached max workers ({max_workers}), job will wait"
            )
            return "WORKERS_BUSY"

        # Check if there's enough VRAM to create a new worker
        model_config = self.model_registry[target_model_id]
        vram_requirement = model_config.get("vram_requirement", 1024)

        # Find GPU with enough VRAM first
        gpu_id = self.gpu_monitor.find_best_gpu(vram_requirement)
        if gpu_id is None:
            # Try to free up VRAM on each GPU systematically until we find one with enough space
            gpu_id = await self._find_gpu_with_freeable_vram(vram_requirement)
        if gpu_id is None:
            logger.info(
                f"No GPU has enough VRAM ({vram_requirement}MB) for model {target_model_id}, job will wait"
            )
            return "NO_VRAM"

        # Create new worker
        worker_id = await self._create_worker_for_model(target_model_id, gpu_id)
        if worker_id:
            return worker_id
        else:
            return "NO_VRAM"  # Failed to create worker, likely VRAM issue

    async def _create_worker_for_model(
        self, model_id: str, gpu_id: int
    ) -> Optional[str]:
        """Create a new worker for the specified model on the given GPU"""
        try:
            model_config = self.model_registry[model_id]
            vram_requirement = model_config.get("vram_requirement", 1024)

            # Allocate VRAM on the target GPU
            if not self.gpu_monitor.allocate_vram(gpu_id, vram_requirement):
                logger.error(
                    f"Failed to allocate {vram_requirement}MB VRAM on GPU {gpu_id} for model {model_id}"
                )
                return None

            # Generate unique worker ID
            worker_count = len(self.worker_assignments.get(model_id, []))
            worker_id = f"worker_{model_id}_{gpu_id}_{worker_count}"

            # Create worker configuration
            worker_config = WorkerConfig(
                worker_id=worker_id, gpu_id=gpu_id, model_config=model_config
            )

            # Create communication queues
            job_queue = mp.Queue(maxsize=100)
            response_queue = mp.Queue()
            control_queue = mp.Queue()
            control_response_queue = mp.Queue()

            # Create and start worker process
            worker_process = mp.Process(
                target=model_worker_process,
                args=(
                    worker_config,
                    job_queue,
                    response_queue,
                    control_queue,
                    control_response_queue,
                ),
                name=worker_id,
            )

            # Store worker information
            self.workers[worker_id] = worker_process
            self.worker_configs[worker_id] = worker_config
            self.worker_queues[worker_id] = job_queue
            self.worker_response_queues[worker_id] = response_queue
            self.worker_control_queues[worker_id] = control_queue
            self.worker_control_response_queues[worker_id] = control_response_queue

            # Initialize worker as available
            self.worker_status[worker_id] = False
            self.worker_last_used[worker_id] = time.time()

            # Track assignment
            if model_id not in self.worker_assignments:
                self.worker_assignments[model_id] = []
            self.worker_assignments[model_id].append(worker_id)

            # Start process
            worker_process.start()
            logger.info(
                f"Created worker {worker_id} for model {model_id} on GPU {gpu_id}"
            )

            return worker_id

        except Exception as e:
            logger.error(f"Error creating worker for model {model_id}: {e}")
            # Deallocate VRAM if worker creation failed
            vram_requirement = model_config.get("vram_requirement", 1024)
            self.gpu_monitor.deallocate_vram(gpu_id, vram_requirement)
            return None

    async def _find_gpu_with_freeable_vram(self, required_vram: int) -> Optional[int]:
        """
        Find a GPU where we can free up enough VRAM by destroying idle workers.

        Args:
            required_vram: Required VRAM in MB

        Returns:
            GPU ID if a suitable GPU is found, None otherwise
        """
        # Get all available GPUs and check each one
        gpu_status = self.gpu_monitor.get_gpu_status()

        for gpu_info in gpu_status:
            gpu_id = gpu_info["id"]
            current_available = self.gpu_monitor.get_gpu_available_vram(gpu_id)

            if current_available >= required_vram:
                # This GPU already has enough VRAM
                return gpu_id

            # Check if we can free up enough VRAM on this GPU
            needed_to_free = required_vram - current_available

            # Find idle workers on this GPU
            idle_workers_on_gpu = []
            current_time = time.time()

            for worker_id, is_busy in self.worker_status.items():
                if not is_busy:
                    worker_config = self.worker_configs[worker_id]
                    if worker_config.gpu_id == gpu_id:
                        last_used = self.worker_last_used.get(worker_id, current_time)
                        idle_time = current_time - last_used

                        # Only consider workers that have been idle for a reasonable time
                        if idle_time > 10:  # 10 seconds
                            worker_vram = worker_config.model_config.get(
                                "vram_requirement", 1024
                            )
                            idle_workers_on_gpu.append(
                                (worker_id, idle_time, worker_vram)
                            )

            # Sort by idle time (longest idle first)
            idle_workers_on_gpu.sort(key=lambda x: x[1], reverse=True)

            # Check if we can free up enough VRAM by destroying idle workers
            potential_freed_vram = 0
            for _, _, worker_vram in idle_workers_on_gpu:
                potential_freed_vram += worker_vram
                if potential_freed_vram >= needed_to_free:
                    # We can free up enough VRAM on this GPU
                    logger.info(
                        f"Can free up {potential_freed_vram}MB VRAM on GPU {gpu_id} (need {needed_to_free}MB)"
                    )

                    # Actually free up the VRAM
                    await self._ensure_vram_available(required_vram, gpu_id)

                    # Verify that we now have enough VRAM
                    final_available = self.gpu_monitor.get_gpu_available_vram(gpu_id)
                    if final_available >= required_vram:
                        return gpu_id
                    break

        logger.info(f"No GPU can be made to have {required_vram}MB VRAM available")
        return None

    async def _ensure_vram_available(
        self, required_vram: int, target_gpu_id: Optional[int] = None
    ):
        """Ensure sufficient VRAM is available on target GPU by destroying idle workers if needed"""
        if target_gpu_id is not None:
            # Check specific GPU
            available_vram = self.gpu_monitor.get_gpu_available_vram(target_gpu_id)

            if available_vram >= required_vram:
                return  # Sufficient VRAM already available on target GPU

            logger.info(
                f"Freeing VRAM on GPU {target_gpu_id}: need {required_vram}MB, have {available_vram}MB"
            )

            # Find idle workers on the target GPU to destroy
            idle_workers = []
            current_time = time.time()

            for worker_id, is_busy in self.worker_status.items():
                if not is_busy:
                    worker_config = self.worker_configs[worker_id]
                    worker_gpu_id = worker_config.gpu_id

                    # Only consider workers on the target GPU
                    if worker_gpu_id == target_gpu_id:
                        last_used = self.worker_last_used.get(worker_id, current_time)
                        idle_time = current_time - last_used

                        # Only consider workers that have been idle for a reasonable time
                        if idle_time > 10:  # 10 seconds
                            idle_workers.append((worker_id, idle_time))

            # Sort by idle time (longest idle first)
            idle_workers.sort(key=lambda x: x[1], reverse=True)

            # Destroy idle workers until we have enough VRAM on target GPU
            for worker_id, _ in idle_workers:
                current_available = self.gpu_monitor.get_gpu_available_vram(
                    target_gpu_id
                )
                if current_available >= required_vram:
                    break

                try:
                    # Get VRAM that will be freed by this worker
                    worker_config = self.worker_configs[worker_id]
                    model_config = worker_config.model_config
                    freed_vram = model_config.get("vram_requirement", 1024)

                    await self._destroy_worker(worker_id)

                    logger.info(
                        f"Destroyed idle worker {worker_id} on GPU {target_gpu_id}, freed {freed_vram}MB VRAM"
                    )

                except Exception as e:
                    logger.error(f"Error destroying idle worker {worker_id}: {e}")
        else:
            logger.warning(
                "_ensure_vram_available called without target_gpu_id - this should not happen in the new per-GPU logic"
            )

    async def _find_worker_for_job(self, job_request: JobRequest) -> Optional[str]:
        """Legacy method - now redirects to the new implementation"""
        return await self._find_or_create_worker_for_job(job_request)

    def _find_available_worker(self, worker_ids: List[str]) -> Optional[str]:
        """
        Find an available (not busy) worker from the given list.

        Args:
            worker_ids: List of worker IDs to check

        Returns:
            Available worker ID or None if all are busy
        """
        for worker_id in worker_ids:
            # Check if worker exists and is not busy
            if worker_id in self.workers and not self.worker_status.get(
                worker_id, False
            ):
                return worker_id
        return None

    def _mark_worker_busy(self, worker_id: str):
        """Mark a worker as busy"""
        self.worker_status[worker_id] = True
        self.worker_last_used[worker_id] = time.time()

    def _mark_worker_available(self, worker_id: str):
        """Mark a worker as available"""
        self.worker_status[worker_id] = False
        self.worker_last_used[worker_id] = time.time()

    def _handle_worker_responses(self):
        """Handle responses from worker processes (runs in separate thread)"""
        while self.running:
            try:
                # Check all response queues
                for worker_id, response_queue in self.worker_response_queues.items():
                    try:
                        callback_id, result = response_queue.get_nowait()
                        logger.info(
                            f"Received response from worker {worker_id}: callback_id={callback_id}, result={result}"
                        )

                        # Find corresponding future and mark worker as available
                        with self.result_lock:
                            if callback_id in self.pending_results:
                                future = self.pending_results.pop(callback_id)

                                # Mark worker as available again
                                self._mark_worker_available(worker_id)
                                logger.info(f"Marked worker {worker_id} as available")

                                # Set result in event loop
                                if self.main_event_loop and not future.done():
                                    self.main_event_loop.call_soon_threadsafe(
                                        future.set_result, result
                                    )
                                    logger.info(
                                        f"Set future result for callback {callback_id}"
                                    )
                                else:
                                    logger.warning(
                                        f"Cannot set future result: main_event_loop={self.main_event_loop}, future.done()={future.done()}"
                                    )
                            else:
                                logger.warning(
                                    f"Callback ID {callback_id} not found in pending results"
                                )

                    except queue.Empty:
                        continue

                time.sleep(0.01)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error handling worker responses: {e}")
                time.sleep(0.1)

    async def _send_control_message(
        self, worker_id: str, msg_type: str, data: Any = None, timeout: float = 10.0
    ) -> WorkerResponse:
        """Send a control message to a worker and wait for response"""
        if worker_id not in self.worker_control_queues:
            raise Exception(f"Worker {worker_id} not found")

        msg = WorkerMessage(msg_type, data)

        # Send message
        self.worker_control_queues[worker_id].put(msg)

        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.worker_control_response_queues[worker_id].get_nowait()
                if response.msg_id == msg.msg_id:
                    return response
            except queue.Empty:
                await asyncio.sleep(0.01)

        raise TimeoutError(f"Timeout waiting for response from worker {worker_id}")

    async def _cleanup_idle_workers(self):
        """Cleanup idle workers periodically"""
        while self.running:
            try:
                current_time = time.time()

                # Find workers that have been idle for a long time
                workers_to_destroy = []

                for worker_id, is_busy in self.worker_status.items():
                    if not is_busy:
                        last_used = self.worker_last_used.get(worker_id, current_time)
                        idle_time = current_time - last_used

                        # Only destroy workers that have been idle for a significant time
                        # and only if we have more than one worker for the model
                        if idle_time > self.idle_cleanup_interval:
                            worker_config = self.worker_configs[worker_id]
                            model_id = worker_config.model_config["model_id"]

                            # Keep at least one worker per model if possible
                            model_workers = self.worker_assignments.get(model_id, [])
                            idle_workers_for_model = [
                                w
                                for w in model_workers
                                if w in self.worker_status and not self.worker_status[w]
                            ]

                            # Only destroy if there are multiple idle workers for this model
                            if len(idle_workers_for_model) > 1:
                                workers_to_destroy.append(worker_id)

                # Destroy excess idle workers
                for worker_id in workers_to_destroy:
                    try:
                        await self._destroy_worker(worker_id)
                        logger.info(f"Cleaned up idle worker {worker_id}")
                    except Exception as e:
                        logger.error(f"Error cleaning up worker {worker_id}: {e}")

                # Wait between cleanup cycles
                await asyncio.sleep(self.idle_cleanup_interval)

            except Exception as e:
                logger.error(f"Error in idle worker cleanup: {e}")
                await asyncio.sleep(10)  # Wait longer on error

    async def _job_queue_cleanup_loop(self):
        """Periodically clean up expired jobs in the job queue"""
        while self.running:
            try:
                await self.job_queue.cleanup_expired_jobs()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in job queue cleanup: {e}")
                await asyncio.sleep(60)

    async def _destroy_worker(self, worker_id: str):
        """Destroy a worker process"""
        if worker_id not in self.workers:
            return

        try:
            # Remove from worker assignments first
            worker_config = self.worker_configs[worker_id]
            model_id = worker_config.model_config["model_id"]
            gpu_id = worker_config.gpu_id
            vram_requirement = worker_config.model_config.get("vram_requirement", 1024)

            if model_id in self.worker_assignments:
                if worker_id in self.worker_assignments[model_id]:
                    self.worker_assignments[model_id].remove(worker_id)

                # If no workers left for this model, remove the entry
                if not self.worker_assignments[model_id]:
                    del self.worker_assignments[model_id]

            # Send shutdown message
            try:
                await self._send_control_message(
                    worker_id, WorkerMessage.Type.SHUTDOWN, timeout=5.0
                )
            except Exception:
                pass  # Ignore timeout/communication errors during shutdown

            # Terminate the process
            process = self.workers.pop(worker_id)
            process.join(timeout=5.0)
            if process.is_alive():
                logger.warning(f"Force terminating worker {worker_id}")
                process.terminate()
                process.join(timeout=2.0)
                if process.is_alive():
                    process.kill()

            # Deallocate VRAM from GPU
            self.gpu_monitor.deallocate_vram(gpu_id, vram_requirement)

            # Cleanup all tracking data
            self.worker_configs.pop(worker_id, None)
            self.worker_queues.pop(worker_id, None)
            self.worker_response_queues.pop(worker_id, None)
            self.worker_control_queues.pop(worker_id, None)
            self.worker_control_response_queues.pop(worker_id, None)
            self.worker_status.pop(worker_id, None)
            self.worker_last_used.pop(worker_id, None)

            logger.info(
                f"Worker {worker_id} destroyed and cleaned up, deallocated {vram_requirement}MB VRAM from GPU {gpu_id}"
            )

        except Exception as e:
            logger.error(f"Error destroying worker {worker_id}: {e}")
