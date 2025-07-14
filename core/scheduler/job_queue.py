import asyncio
import logging
import uuid
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .database_manager import DatabaseManager
from .database_models import JobModel
from .database_models import JobStatus as DBJobStatus

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRequest:
    """Represents a job request in the system"""

    def __init__(
        self,
        feature: str,
        inputs: Dict[str, Any],
        model_preference: Optional[str] = None,
        priority: int = 1,
        timeout_seconds: int = 3600,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.job_id = str(uuid.uuid4())
        self.feature = feature
        self.inputs = inputs
        self.model_preference = model_preference
        self.priority = priority
        self.timeout_seconds = timeout_seconds
        self.metadata = metadata or {}

        # Status tracking
        self.status = JobStatus.QUEUED
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.assigned_model: Optional[str] = None

        # Retry tracking
        self.retry_count: int = 0
        self.max_retries: int = 4  # Default max retries
        self.last_retry_at: Optional[datetime] = None
        self.retry_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRequest":
        """Create JobRequest from dictionary representation"""
        job = cls(
            feature=data["feature"],
            inputs=data["inputs"],
            model_preference=data.get("model_preference"),
            priority=data.get("priority", 1),
            timeout_seconds=data.get("timeout_seconds", 3600),
            metadata=data.get("metadata", {}),
        )

        # Restore additional fields
        job.job_id = data["job_id"]
        job.status = JobStatus(data["status"])
        job.progress = data.get("progress", 0.0)
        job.assigned_model = data.get("assigned_model")

        # Parse datetime fields
        job.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.fromisoformat(data["completed_at"])

        job.result = data.get("result")
        job.error = data.get("error")

        # Restore retry tracking
        job.retry_count = data.get("retry_count", 0)
        job.max_retries = data.get("max_retries", 5)
        if data.get("last_retry_at"):
            job.last_retry_at = datetime.fromisoformat(data["last_retry_at"])
        job.retry_reason = data.get("retry_reason")

        return job

    @classmethod
    def from_job_model(cls, job_model: "JobModel") -> "JobRequest":
        """Create JobRequest from JobModel database object"""
        job = cls(
            feature=job_model.feature,  # type: ignore
            inputs=job_model.inputs,  # type: ignore
            model_preference=job_model.model_preference,  # type: ignore
            priority=job_model.priority,  # type: ignore
            timeout_seconds=job_model.timeout_seconds,  # type: ignore
            metadata=job_model.job_metadata or {},  # type: ignore
        )

        # Restore additional fields
        job.job_id = job_model.job_id  # type: ignore
        job.status = JobStatus(job_model.status)  # type: ignore
        job.progress = job_model.progress  # type: ignore
        job.assigned_model = job_model.assigned_model  # type: ignore

        # Restore datetime fields
        job.created_at = job_model.created_at  # type: ignore
        job.started_at = job_model.started_at  # type: ignore
        job.completed_at = job_model.completed_at  # type: ignore

        job.result = job_model.result  # type: ignore
        job.error = job_model.error  # type: ignore

        # Restore retry tracking
        job.retry_count = job_model.retry_count  # type: ignore
        job.max_retries = job_model.max_retries  # type: ignore
        job.last_retry_at = job_model.last_retry_at  # type: ignore
        job.retry_reason = job_model.retry_reason  # type: ignore

        return job

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation"""
        return {
            "job_id": self.job_id,
            "feature": self.feature,
            "inputs": self.inputs,
            "model_preference": self.model_preference,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value,
            "progress": self.progress,
            "assigned_model": self.assigned_model,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_retry_at": self.last_retry_at.isoformat()
            if self.last_retry_at
            else None,
            "retry_reason": self.retry_reason,
        }

    def is_expired(self) -> bool:
        """Check if job has exceeded timeout"""
        if self.started_at and self.status == JobStatus.PROCESSING:
            elapsed = datetime.utcnow() - self.started_at
            return elapsed.total_seconds() > self.timeout_seconds
        return False

    def is_waiting_too_long(self) -> bool:
        """Check if job has been waiting in queue for more than 1 hour (impossible job detection)"""
        elapsed = datetime.utcnow() - self.created_at
        return elapsed.total_seconds() > 3600  # 1 hour in seconds

    def mark_started(self, model_id: str):
        """Mark job as started"""
        self.status = JobStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.assigned_model = model_id

    def mark_completed(self, result: Dict[str, Any]):
        """Mark job as completed"""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        self.progress = 1.0

    def mark_failed(self, error: str):
        """Mark job as failed"""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error

    def mark_cancelled(self):
        """Mark job as cancelled"""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()


class JobQueue:
    """Database-backed job queue with timeout handling and persistence"""

    def __init__(
        self,
        max_size: int = 1000,
        database_url: Optional[str] = None,
        max_completed_jobs: int = 1000,
        # Legacy parameters for backward compatibility
        persistence_file: Optional[str] = None,
        persistence_interval: int = 30,
    ):
        self.max_size = max_size
        self.max_completed_jobs = max_completed_jobs

        # Initialize database manager
        self.db_manager = DatabaseManager(database_url=database_url)

        # In-memory caches for performance (synchronized with database)
        self._queue_cache = deque()
        self._processing_cache: Dict[str, JobRequest] = {}
        self._completed_cache: Dict[str, JobRequest] = {}

        # Thread-safe lock for cache operations
        self._cache_lock = asyncio.Lock()

        # Background task management
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        # Load existing jobs from database on initialization
        self._load_from_database()

    def _load_from_database(self):
        """Load existing jobs from database into memory caches"""
        try:
            # Load queued jobs
            queued_jobs = self.db_manager.get_queued_jobs_ordered()
            for job_model in queued_jobs:
                job_request = JobRequest.from_job_model(job_model)
                self._queue_cache.append(job_request)

            # Load processing jobs
            processing_jobs = self.db_manager.get_jobs_by_status(DBJobStatus.PROCESSING)
            for job_model in processing_jobs:
                job_request = JobRequest.from_job_model(job_model)
                self._processing_cache[job_request.job_id] = job_request

            # Load recent completed jobs
            completed_jobs = self.db_manager.get_jobs_by_status(DBJobStatus.COMPLETED)
            failed_jobs = self.db_manager.get_jobs_by_status(DBJobStatus.FAILED)
            cancelled_jobs = self.db_manager.get_jobs_by_status(DBJobStatus.CANCELLED)

            all_completed = completed_jobs + failed_jobs + cancelled_jobs

            # Sort by completion time and keep only the most recent
            all_completed.sort(
                key=lambda x: x.completed_at or datetime.min,  # type: ignore
                reverse=True,
            )

            for job_model in all_completed[: self.max_completed_jobs]:
                job_request = JobRequest.from_job_model(job_model)
                self._completed_cache[job_request.job_id] = job_request

            logger.info(
                f"Loaded {len(self._queue_cache)} queued, {len(self._processing_cache)} processing, "
                f"and {len(self._completed_cache)} completed jobs from database"
            )

        except Exception as e:
            logger.error(f"Failed to load jobs from database: {e}")
            # Initialize empty caches on error
            self._queue_cache.clear()
            self._processing_cache.clear()
            self._completed_cache.clear()

    async def start_persistence(self):
        """Start background tasks for database synchronization"""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started database-backed job queue")

    async def stop_persistence(self):
        """Stop background tasks"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Close database connections
        self.db_manager.close()
        logger.info("Stopped database-backed job queue")

    async def _periodic_cleanup(self):
        """Periodically clean up old jobs and sync with database"""
        while self._running:
            try:
                # Clean up old completed jobs in database
                cleaned_count = self.db_manager.cleanup_old_jobs(
                    self.max_completed_jobs
                )

                # Clean up expired jobs
                await self.cleanup_expired_jobs()

                # Sync completed cache with database if needed
                async with self._cache_lock:
                    if len(self._completed_cache) > self.max_completed_jobs:
                        # Keep only the most recent completed jobs in cache
                        sorted_completed = sorted(
                            self._completed_cache.values(),
                            key=lambda job: job.completed_at or datetime.min,
                            reverse=True,
                        )

                        jobs_to_keep = sorted_completed[: self.max_completed_jobs]
                        self._completed_cache = {
                            job.job_id: job for job in jobs_to_keep
                        }

                await asyncio.sleep(60)  # Run cleanup every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)

    async def enqueue(self, job: JobRequest) -> str:
        """Add job to queue"""
        async with self._cache_lock:
            if len(self._queue_cache) >= self.max_size:
                raise Exception("Job queue is full")

            # Save to database first
            if not self.db_manager.save_job(job):
                raise Exception("Failed to save job to database")

            # Add to cache
            self._queue_cache.append(job)
            self._sort_queue_cache()

            logger.info(f"Enqueued job {job.job_id} for feature {job.feature}")
            return job.job_id

    async def enqueue_front(self, job: JobRequest):
        """Add job to front of queue (for jobs that couldn't be processed due to resources)"""
        async with self._cache_lock:
            if len(self._queue_cache) >= self.max_size:
                raise Exception("Job queue is full")

            # Save to database first
            if not self.db_manager.save_job(job):
                raise Exception("Failed to save job to database")

            # Add to front of cache
            self._queue_cache.appendleft(job)
            logger.info(
                f"Re-enqueued job {job.job_id} to front of queue for feature {job.feature}"
            )

    async def requeue_job(self, job_id: str):
        """Move a job from processing back to front of queue"""
        async with self._cache_lock:
            if job_id in self._processing_cache:
                job = self._processing_cache[job_id]

                # Reset job status
                job.status = JobStatus.QUEUED
                job.started_at = None
                job.assigned_model = None
                job.progress = 0.0

                # Save updated job to database
                if not self.db_manager.save_job(job):
                    logger.error(f"Failed to save requeued job {job_id} to database")
                    return False

                # Move from processing to front of queue
                del self._processing_cache[job_id]
                self._queue_cache.appendleft(job)

                logger.info(f"Requeued job {job_id} to front of queue")
                return True
            return False

    async def dequeue(self) -> Optional[JobRequest]:
        """Get next job from queue"""
        async with self._cache_lock:
            if not self._queue_cache:
                return None

            job = self._queue_cache.popleft()

            # Move to processing cache
            self._processing_cache[job.job_id] = job

            return job

    async def mark_job_started(self, job_id: str, model_id: str):
        """Mark job as started processing"""
        async with self._cache_lock:
            if job_id in self._processing_cache:
                job = self._processing_cache[job_id]
                job.mark_started(model_id)

                # Save to database
                if not self.db_manager.save_job(job):
                    logger.error(f"Failed to save started job {job_id} to database")
                else:
                    logger.info(
                        f"Job {job_id} started processing with model {model_id}"
                    )

    async def complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark job as completed"""
        async with self._cache_lock:
            if job_id in self._processing_cache:
                job = self._processing_cache[job_id]
                job.mark_completed(result)

                # Save to database
                if not self.db_manager.save_job(job):
                    logger.error(f"Failed to save completed job {job_id} to database")

                # Move from processing to completed
                del self._processing_cache[job_id]
                self._completed_cache[job_id] = job

                logger.info(f"Completed job {job_id}")

    async def fail_job(self, job_id: str, error: str):
        """Mark job as failed"""
        async with self._cache_lock:
            if job_id in self._processing_cache:
                job = self._processing_cache[job_id]
                job.mark_failed(error)

                # Save to database
                if not self.db_manager.save_job(job):
                    logger.error(f"Failed to save failed job {job_id} to database")

                # Move from processing to completed
                del self._processing_cache[job_id]
                self._completed_cache[job_id] = job

                logger.error(f"Failed job {job_id}: {error}")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's still queued"""
        async with self._cache_lock:
            # Check if job is in queue
            for i, job in enumerate(self._queue_cache):
                if job.job_id == job_id:
                    job.mark_cancelled()

                    # Save to database
                    if not self.db_manager.save_job(job):
                        logger.error(
                            f"Failed to save cancelled job {job_id} to database"
                        )

                    # Move from queue to completed
                    del self._queue_cache[i]
                    self._completed_cache[job_id] = job

                    logger.info(f"Cancelled job {job_id}")
                    return True

            # Check if job is processing (can't cancel)
            if job_id in self._processing_cache:
                logger.warning(f"Cannot cancel job {job_id}: already processing")
                return False

            return False

    async def get_job(self, job_id: str) -> Optional[JobRequest]:
        """Get job by ID"""
        async with self._cache_lock:
            # Check processing jobs first
            if job_id in self._processing_cache:
                return self._processing_cache[job_id]

            # Check completed jobs
            if job_id in self._completed_cache:
                return self._completed_cache[job_id]

            # Check queued jobs
            for job in self._queue_cache:
                if job.job_id == job_id:
                    return job

            # If not in cache, try database
            job_model = self.db_manager.get_job(job_id)
            if job_model:
                return JobRequest.from_job_model(job_model)

            return None

    async def update_job_progress(self, job_id: str, progress: float):
        """Update job progress"""
        async with self._cache_lock:
            if job_id in self._processing_cache:
                job = self._processing_cache[job_id]
                job.progress = min(1.0, max(0.0, progress))

                # Save to database
                if not self.db_manager.save_job(job):
                    logger.error(
                        f"Failed to save job progress for {job_id} to database"
                    )

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self._cache_lock:
            return {
                "queued_jobs": len(self._queue_cache),
                "processing_jobs": len(self._processing_cache),
                "completed_jobs": len(self._completed_cache),
                "max_queue_size": self.max_size,
                "queue_utilization": len(self._queue_cache) / self.max_size,
            }

    async def get_jobs_by_status(self, status: JobStatus) -> List[JobRequest]:
        """Get all jobs with specified status"""
        async with self._cache_lock:
            jobs = []

            if status == JobStatus.QUEUED:
                jobs.extend(self._queue_cache)
            elif status == JobStatus.PROCESSING:
                jobs.extend(self._processing_cache.values())
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                jobs.extend(
                    [
                        job
                        for job in self._completed_cache.values()
                        if job.status == status
                    ]
                )

            return jobs

    async def cleanup_expired_jobs(self):
        """Clean up expired processing jobs"""
        async with self._cache_lock:
            expired_jobs = []

            for job_id, job in list(self._processing_cache.items()):
                if job.is_expired():
                    expired_jobs.append(job_id)

            for job_id in expired_jobs:
                job = self._processing_cache[job_id]
                job.mark_failed("Job timeout exceeded")

                # Save to database
                if not self.db_manager.save_job(job):
                    logger.error(f"Failed to save expired job {job_id} to database")

                # Move from processing to completed
                del self._processing_cache[job_id]
                self._completed_cache[job_id] = job

                logger.warning(f"Job {job_id} expired after timeout")

    def _sort_queue_cache(self):
        """Sort queue cache by priority (higher priority first)"""
        self._queue_cache = deque(
            sorted(self._queue_cache, key=lambda job: (-job.priority, job.created_at))
        )

    # Legacy methods for backward compatibility
    async def _cleanup_completed_jobs(self):
        """Legacy method - now handled by periodic cleanup"""
        pass

    def _save_state(self):
        """Legacy method - persistence is now handled by database"""
        pass

    def _load_state(self):
        """Legacy method - loading is now handled by database"""
        pass

    async def _periodic_persistence(self):
        """Legacy method - persistence is now handled by database"""
        pass
