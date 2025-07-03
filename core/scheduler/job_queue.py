import asyncio
import json
import logging
import os
import uuid
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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
    """Priority-based job queue with timeout handling and persistence"""

    def __init__(
        self,
        max_size: int = 1000,
        persistence_file: Optional[str] = None,
        persistence_interval: int = 30,
    ):
        self.max_size = max_size
        self.persistence_file = persistence_file
        self.persistence_interval = persistence_interval
        self._queue = deque()
        self._processing_jobs: Dict[str, JobRequest] = {}
        self._completed_jobs: Dict[str, JobRequest] = {}
        self._lock = asyncio.Lock()
        self._max_completed_jobs = 1000
        self._persistence_task: Optional[asyncio.Task] = None
        self._running = False

        # Load state from disk if persistence file exists
        if self.persistence_file and os.path.exists(self.persistence_file):
            try:
                self._load_state()
                logger.info(f"Loaded job queue state from {self.persistence_file}")
            except Exception as e:
                logger.error(f"Failed to load job queue state: {e}")

    async def start_persistence(self):
        """Start the periodic persistence task"""
        if self.persistence_file and not self._persistence_task:
            self._running = True
            self._persistence_task = asyncio.create_task(self._periodic_persistence())
            logger.info(f"Started job queue persistence to {self.persistence_file}")

    async def stop_persistence(self):
        """Stop the periodic persistence task"""
        self._running = False
        if self._persistence_task:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass
            self._persistence_task = None

        # Final save
        if self.persistence_file:
            try:
                self._save_state()
                logger.info("Final job queue state saved")
            except Exception as e:
                logger.error(f"Failed to save final job queue state: {e}")

    def _save_state(self):
        """Save current queue state to disk"""
        if not self.persistence_file:
            return

        try:
            state = {
                "queue": [job.to_dict() for job in self._queue],
                "processing_jobs": {
                    job_id: job.to_dict()
                    for job_id, job in self._processing_jobs.items()
                },
                "completed_jobs": {
                    job_id: job.to_dict()
                    for job_id, job in self._completed_jobs.items()
                },
                "saved_at": datetime.utcnow().isoformat(),
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)

            # Write to temporary file first, then rename for atomic operation
            temp_file = f"{self.persistence_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2)

            os.rename(temp_file, self.persistence_file)

        except Exception as e:
            logger.error(f"Failed to save job queue state: {e}")
            raise

    def _load_state(self):
        """Load queue state from disk"""
        if not self.persistence_file or not os.path.exists(self.persistence_file):
            return

        try:
            with open(self.persistence_file, "r") as f:
                state = json.load(f)

            # Restore queued jobs
            self._queue = deque()
            for job_data in state.get("queue", []):
                job = JobRequest.from_dict(job_data)
                self._queue.append(job)

            # Restore processing jobs
            self._processing_jobs = {}
            for job_id, job_data in state.get("processing_jobs", {}).items():
                job = JobRequest.from_dict(job_data)
                self._processing_jobs[job_id] = job

            # Restore completed jobs
            self._completed_jobs = {}
            for job_id, job_data in state.get("completed_jobs", {}).items():
                job = JobRequest.from_dict(job_data)
                self._completed_jobs[job_id] = job

            logger.info(
                f"Restored {len(self._queue)} queued, {len(self._processing_jobs)} processing, "
                f"and {len(self._completed_jobs)} completed jobs"
            )

        except Exception as e:
            logger.error(f"Failed to load job queue state: {e}")
            raise

    async def _periodic_persistence(self):
        """Periodically save queue state to disk"""
        while self._running:
            try:
                await asyncio.sleep(self.persistence_interval)
                if self._running:  # Check again after sleep
                    async with self._lock:
                        self._save_state()
                    logger.debug("Periodic job queue state saved")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic persistence: {e}")
                await asyncio.sleep(self.persistence_interval)

    async def enqueue(self, job: JobRequest) -> str:
        """Add job to queue"""
        async with self._lock:
            if len(self._queue) >= self.max_size:
                raise Exception("Job queue is full")

            self._queue.append(job)
            self._sort_queue()
            logger.info(f"Enqueued job {job.job_id} for feature {job.feature}")
            return job.job_id

    async def enqueue_front(self, job: JobRequest):
        """Add job to front of queue (for jobs that couldn't be processed due to resources)"""
        async with self._lock:
            if len(self._queue) >= self.max_size:
                raise Exception("Job queue is full")

            self._queue.appendleft(job)
            logger.info(
                f"Re-enqueued job {job.job_id} to front of queue for feature {job.feature}"
            )

    async def requeue_job(self, job_id: str):
        """Move a job from processing back to front of queue"""
        async with self._lock:
            if job_id in self._processing_jobs:
                job = self._processing_jobs[job_id]
                # Reset job status to queued
                job.status = JobStatus.QUEUED
                job.started_at = None
                job.assigned_model = None
                job.progress = 0.0

                # Remove from processing and add to front of queue
                del self._processing_jobs[job_id]
                self._queue.appendleft(job)
                logger.info(f"Requeued job {job_id} to front of queue")
                return True
            return False

    async def dequeue(self) -> Optional[JobRequest]:
        """Get next job from queue"""
        async with self._lock:
            if not self._queue:
                return None

            job = self._queue.popleft()
            self._processing_jobs[job.job_id] = job
            return job

    async def mark_job_started(self, job_id: str, model_id: str):
        """Mark job as started processing"""
        async with self._lock:
            if job_id in self._processing_jobs:
                self._processing_jobs[job_id].mark_started(model_id)
                logger.info(f"Job {job_id} started processing with model {model_id}")

    async def complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark job as completed"""
        async with self._lock:
            if job_id in self._processing_jobs:
                job = self._processing_jobs[job_id]
                job.mark_completed(result)
                del self._processing_jobs[job_id]
                self._completed_jobs[job_id] = job
                await self._cleanup_completed_jobs()
                logger.info(f"Completed job {job_id}")

    async def fail_job(self, job_id: str, error: str):
        """Mark job as failed"""
        async with self._lock:
            if job_id in self._processing_jobs:
                job = self._processing_jobs[job_id]
                job.mark_failed(error)
                del self._processing_jobs[job_id]
                self._completed_jobs[job_id] = job
                await self._cleanup_completed_jobs()
                logger.error(f"Failed job {job_id}: {error}")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's still queued"""
        async with self._lock:
            # Check if job is in queue
            for i, job in enumerate(self._queue):
                if job.job_id == job_id:
                    job.mark_cancelled()
                    del self._queue[i]
                    self._completed_jobs[job_id] = job
                    logger.info(f"Cancelled job {job_id}")
                    return True

            # Check if job is processing (can't cancel)
            if job_id in self._processing_jobs:
                logger.warning(f"Cannot cancel job {job_id}: already processing")
                return False

            return False

    async def get_job(self, job_id: str) -> Optional[JobRequest]:
        """Get job by ID"""
        async with self._lock:
            # Check processing jobs
            if job_id in self._processing_jobs:
                return self._processing_jobs[job_id]

            # Check completed jobs
            if job_id in self._completed_jobs:
                return self._completed_jobs[job_id]

            # Check queued jobs
            for job in self._queue:
                if job.job_id == job_id:
                    return job

            return None

    async def update_job_progress(self, job_id: str, progress: float):
        """Update job progress"""
        async with self._lock:
            if job_id in self._processing_jobs:
                self._processing_jobs[job_id].progress = min(1.0, max(0.0, progress))

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queued_jobs": len(self._queue),
                "processing_jobs": len(self._processing_jobs),
                "completed_jobs": len(self._completed_jobs),
                "max_queue_size": self.max_size,
                "queue_utilization": len(self._queue) / self.max_size,
            }

    async def get_jobs_by_status(self, status: JobStatus) -> List[JobRequest]:
        """Get all jobs with specified status"""
        async with self._lock:
            jobs = []

            if status == JobStatus.QUEUED:
                jobs.extend(self._queue)
            elif status == JobStatus.PROCESSING:
                jobs.extend(self._processing_jobs.values())
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                jobs.extend(
                    [
                        job
                        for job in self._completed_jobs.values()
                        if job.status == status
                    ]
                )

            return jobs

    async def cleanup_expired_jobs(self):
        """Clean up expired processing jobs"""
        async with self._lock:
            expired_jobs = []
            for job_id, job in list(self._processing_jobs.items()):
                if job.is_expired():
                    expired_jobs.append(job_id)

            for job_id in expired_jobs:
                job = self._processing_jobs[job_id]
                job.mark_failed("Job timeout exceeded")
                del self._processing_jobs[job_id]
                self._completed_jobs[job_id] = job
                logger.warning(f"Job {job_id} expired after timeout")

            if expired_jobs:
                await self._cleanup_completed_jobs()

    def _sort_queue(self):
        """Sort queue by priority (higher priority first)"""
        self._queue = deque(
            sorted(self._queue, key=lambda job: (-job.priority, job.created_at))
        )

    async def _cleanup_completed_jobs(self):
        """Remove old completed jobs to prevent memory buildup"""
        if len(self._completed_jobs) > self._max_completed_jobs:
            # Sort by completion time and keep only the most recent
            sorted_jobs = sorted(
                self._completed_jobs.values(),
                key=lambda job: job.completed_at or datetime.min,
                reverse=True,
            )

            jobs_to_keep = sorted_jobs[: self._max_completed_jobs]
            self._completed_jobs = {job.job_id: job for job in jobs_to_keep}

            removed_count = len(sorted_jobs) - len(jobs_to_keep)
            logger.info(f"Cleaned up {removed_count} old completed jobs")
