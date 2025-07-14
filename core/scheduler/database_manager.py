"""
Database manager for job queue persistence using SQLAlchemy.
Handles connection management, transactions, and CRUD operations.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, asc, create_engine, desc, func, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .database_models import Base, JobModel, JobStatus

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and operations for job queue persistence.
    Provides thread-safe operations with proper transaction handling.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_pre_ping: bool = True,
    ):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL. If None, uses SQLite with file storage.
            echo: Whether to echo SQL statements (for debugging)
            pool_size: Number of connections to maintain in the pool
            max_overflow: Maximum number of connections to create beyond pool_size
            pool_pre_ping: Whether to validate connections before use
        """
        self.database_url = database_url or self._get_default_database_url()
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_pre_ping = pool_pre_ping

        # Initialize engine and session factory
        self.engine = None
        self.SessionLocal = None
        self._initialized = False

        # Initialize the database
        self._initialize_database()

    def _get_default_database_url(self) -> str:
        """Get default database URL for SQLite."""
        # Create data directory if it doesn't exist
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        # Use SQLite with file storage
        db_path = os.path.join(data_dir, "job_queue.db")
        return f"sqlite:///{db_path}"

    def _initialize_database(self):
        """Initialize database engine and create tables if necessary."""
        try:
            # Create engine based on database type
            if self.database_url.startswith("sqlite"):
                # SQLite specific configuration
                self.engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 30,
                    },
                )
            else:
                # PostgreSQL/MySQL configuration
                self.engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_pre_ping=self.pool_pre_ping,
                    pool_recycle=3600,  # Recycle connections after 1 hour
                )

            # Create session factory
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
            )

            # Create tables
            Base.metadata.create_all(bind=self.engine)

            self._initialized = True
            logger.info(f"Database initialized successfully: {self.database_url}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions with automatic cleanup.

        Usage:
            with db_manager.get_session() as session:
                # Use session for database operations
                pass
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized")

        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def save_job(self, job_request: "JobRequest") -> bool:
        """
        Save or update a job in the database.

        Args:
            job_request: JobRequest instance to save

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                # Check if job already exists
                existing_job = (
                    session.query(JobModel).filter_by(job_id=job_request.job_id).first()
                )

                if existing_job:
                    # Update existing job
                    existing_job.update_from_job_request(job_request)
                    logger.debug(f"Updated job {job_request.job_id} in database")
                else:
                    # Create new job
                    new_job = JobModel.from_job_request(job_request)
                    session.add(new_job)
                    logger.debug(f"Created new job {job_request.job_id} in database")

                session.commit()
                return True

        except SQLAlchemyError as e:
            logger.error(f"Database error saving job {job_request.job_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving job {job_request.job_id}: {e}")
            return False

    def get_job(self, job_id: str) -> Optional[JobModel]:
        """
        Get a job from the database by ID.

        Args:
            job_id: Job ID to retrieve

        Returns:
            JobModel instance or None if not found
        """
        try:
            with self.get_session() as session:
                job = session.query(JobModel).filter_by(job_id=job_id).first()
                if job:
                    # Detach from session to avoid lazy loading issues
                    session.expunge(job)
                return job

        except SQLAlchemyError as e:
            logger.error(f"Database error getting job {job_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting job {job_id}: {e}")
            return None

    def get_jobs_by_status(self, status: JobStatus) -> List[JobModel]:
        """
        Get all jobs with a specific status.

        Args:
            status: Job status to filter by

        Returns:
            List of JobModel instances
        """
        try:
            with self.get_session() as session:
                jobs = session.query(JobModel).filter_by(status=status.value).all()
                # Detach from session
                for job in jobs:
                    session.expunge(job)
                return jobs

        except SQLAlchemyError as e:
            logger.error(f"Database error getting jobs by status {status}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting jobs by status {status}: {e}")
            return []

    def get_queued_jobs_ordered(self) -> List[JobModel]:
        """
        Get all queued jobs ordered by priority and creation time (FIFO).

        Returns:
            List of JobModel instances ordered for FIFO processing
        """
        try:
            with self.get_session() as session:
                jobs = (
                    session.query(JobModel)
                    .filter_by(status=JobStatus.QUEUED.value)
                    .order_by(desc(JobModel.priority), asc(JobModel.created_at))
                    .all()
                )
                # Detach from session
                for job in jobs:
                    session.expunge(job)
                return jobs

        except SQLAlchemyError as e:
            logger.error(f"Database error getting queued jobs: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting queued jobs: {e}")
            return []

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from the database.

        Args:
            job_id: Job ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                job = session.query(JobModel).filter_by(job_id=job_id).first()
                if job:
                    session.delete(job)
                    session.commit()
                    logger.debug(f"Deleted job {job_id} from database")
                    return True
                return False

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting job {job_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting job {job_id}: {e}")
            return False

    def cleanup_old_jobs(self, max_completed_jobs: int = 1000) -> int:
        """
        Clean up old completed jobs to prevent database bloat.

        Args:
            max_completed_jobs: Maximum number of completed jobs to keep

        Returns:
            Number of jobs cleaned up
        """
        try:
            with self.get_session() as session:
                # Get count of completed jobs
                completed_statuses = [
                    JobStatus.COMPLETED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                ]

                total_completed = (
                    session.query(func.count(JobModel.job_id))
                    .filter(JobModel.status.in_(completed_statuses))
                    .scalar()
                )

                if total_completed <= max_completed_jobs:
                    return 0  # No cleanup needed

                # Get IDs of jobs to delete (keep most recent)
                jobs_to_delete = (
                    session.query(JobModel.job_id)
                    .filter(JobModel.status.in_(completed_statuses))
                    .order_by(desc(JobModel.completed_at))
                    .offset(max_completed_jobs)
                    .all()
                )

                job_ids_to_delete = [job.job_id for job in jobs_to_delete]

                # Delete old jobs
                if job_ids_to_delete:
                    deleted_count = (
                        session.query(JobModel)
                        .filter(JobModel.job_id.in_(job_ids_to_delete))
                        .delete(synchronize_session=False)
                    )
                    session.commit()
                    logger.info(f"Cleaned up {deleted_count} old completed jobs")
                    return deleted_count

                return 0

        except SQLAlchemyError as e:
            logger.error(f"Database error cleaning up old jobs: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error cleaning up old jobs: {e}")
            return 0

    def get_queue_statistics(self) -> Dict[str, Any]:
        """
        Get queue statistics from the database.

        Returns:
            Dictionary containing queue statistics
        """
        try:
            with self.get_session() as session:
                # Count jobs by status
                status_counts = {}
                for status in JobStatus:
                    count = (
                        session.query(func.count(JobModel.job_id))
                        .filter_by(status=status.value)
                        .scalar()
                    )
                    status_counts[status.value] = count or 0

                # Get oldest queued job
                oldest_queued = (
                    session.query(JobModel.created_at)
                    .filter_by(status=JobStatus.QUEUED.value)
                    .order_by(asc(JobModel.created_at))
                    .first()
                )

                oldest_queued_time = oldest_queued[0] if oldest_queued else None

                # Calculate queue wait time
                max_wait_time = 0
                if oldest_queued_time:
                    max_wait_time = (
                        datetime.utcnow() - oldest_queued_time
                    ).total_seconds()

                return {
                    "queued_jobs": status_counts.get("queued", 0),
                    "processing_jobs": status_counts.get("processing", 0),
                    "completed_jobs": status_counts.get("completed", 0),
                    "failed_jobs": status_counts.get("failed", 0),
                    "cancelled_jobs": status_counts.get("cancelled", 0),
                    "total_jobs": sum(status_counts.values()),
                    "max_wait_time_seconds": max_wait_time,
                    "oldest_queued_time": oldest_queued_time.isoformat()
                    if oldest_queued_time
                    else None,
                }

        except SQLAlchemyError as e:
            logger.error(f"Database error getting queue statistics: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting queue statistics: {e}")
            return {}

    def get_expired_jobs(self) -> List[JobModel]:
        """
        Get jobs that have exceeded their timeout while processing.

        Returns:
            List of expired JobModel instances
        """
        try:
            with self.get_session() as session:
                current_time = datetime.utcnow()

                # Find processing jobs that have exceeded their timeout
                expired_jobs = (
                    session.query(JobModel)
                    .filter(
                        and_(
                            JobModel.status == JobStatus.PROCESSING.value,
                            JobModel.started_at.isnot(None),
                            func.julianday(current_time)
                            - func.julianday(JobModel.started_at)
                            > JobModel.timeout_seconds
                            / 86400.0,  # Convert seconds to days
                        )
                    )
                    .all()
                )

                # Detach from session
                for job in expired_jobs:
                    session.expunge(job)

                return expired_jobs

        except SQLAlchemyError as e:
            logger.error(f"Database error getting expired jobs: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting expired jobs: {e}")
            return []

    def bulk_save_jobs(self, job_requests: List["JobRequest"]) -> bool:
        """
        Save multiple jobs in a single transaction.

        Args:
            job_requests: List of JobRequest instances to save

        Returns:
            True if all jobs saved successfully, False otherwise
        """
        if not job_requests:
            return True

        try:
            with self.get_session() as session:
                job_models = []
                for job_request in job_requests:
                    # Check if job already exists
                    existing_job = (
                        session.query(JobModel)
                        .filter_by(job_id=job_request.job_id)
                        .first()
                    )

                    if existing_job:
                        existing_job.update_from_job_request(job_request)
                    else:
                        new_job = JobModel.from_job_request(job_request)
                        session.add(new_job)

                session.commit()
                logger.debug(f"Bulk saved {len(job_requests)} jobs to database")
                return True

        except SQLAlchemyError as e:
            logger.error(f"Database error bulk saving jobs: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error bulk saving jobs: {e}")
            return False

    def health_check(self) -> bool:
        """
        Check if database connection is healthy.

        Returns:
            True if database is accessible, False otherwise
        """
        try:
            with self.get_session() as session:
                # Simple query to test connection
                session.execute(text("SELECT 1"))
                return True

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def close(self):
        """Close database connections and cleanup resources."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
