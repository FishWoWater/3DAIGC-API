"""
Database models for job queue persistence using SQLAlchemy.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import TypeDecorator

Base = declarative_base()


class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JSONType(TypeDecorator):
    """Custom SQLAlchemy type for JSON fields that works with multiple databases."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value


class JobModel(Base):
    """SQLAlchemy model for persisting job requests."""

    __tablename__ = "jobs"

    # Primary key
    job_id = Column(String(36), primary_key=True)

    # Job definition
    feature = Column(String(100), nullable=False, index=True)
    inputs = Column(JSONType, nullable=False)
    model_preference = Column(String(100), nullable=True)
    priority = Column(Integer, nullable=False, default=1, index=True)
    timeout_seconds = Column(Integer, nullable=False, default=3600)
    job_metadata = Column(JSONType, nullable=True)

    # Status tracking
    status = Column(String(20), nullable=False, default="queued", index=True)
    progress = Column(Float, nullable=False, default=0.0)
    assigned_model = Column(String(100), nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Results
    result = Column(JSONType, nullable=True)
    error = Column(Text, nullable=True)

    # Retry tracking
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=4)
    last_retry_at = Column(DateTime, nullable=True)
    retry_reason = Column(String(500), nullable=True)

    # Add indexes for common queries
    __table_args__ = (
        Index("idx_jobs_status_created", "status", "created_at"),
        Index("idx_jobs_status_priority", "status", "priority", "created_at"),
        Index("idx_jobs_feature_status", "feature", "status"),
        Index("idx_jobs_completed_at", "completed_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert database model to dictionary."""
        return {
            "job_id": self.job_id,
            "feature": self.feature,
            "inputs": self.inputs,
            "model_preference": self.model_preference,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.job_metadata or {},
            "status": self.status,
            "progress": self.progress,
            "assigned_model": self.assigned_model,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat()
            if self.started_at is not None
            else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at is not None
            else None,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_retry_at": self.last_retry_at.isoformat()
            if self.last_retry_at is not None
            else None,
            "retry_reason": self.retry_reason,
        }

    @classmethod
    def from_job_request(cls, job_request) -> "JobModel":
        """Create JobModel from JobRequest instance."""
        return cls(
            job_id=job_request.job_id,
            feature=job_request.feature,
            inputs=job_request.inputs,
            model_preference=job_request.model_preference,
            priority=job_request.priority,
            timeout_seconds=job_request.timeout_seconds,
            job_metadata=job_request.metadata,
            status=job_request.status.value,
            progress=job_request.progress,
            assigned_model=job_request.assigned_model,
            created_at=job_request.created_at,
            started_at=job_request.started_at,
            completed_at=job_request.completed_at,
            result=job_request.result,
            error=job_request.error,
            retry_count=job_request.retry_count,
            max_retries=job_request.max_retries,
            last_retry_at=job_request.last_retry_at,
            retry_reason=job_request.retry_reason,
        )

    def update_from_job_request(self, job_request):
        """Update model fields from JobRequest instance."""
        self.feature = job_request.feature
        self.inputs = job_request.inputs
        self.model_preference = job_request.model_preference
        self.priority = job_request.priority
        self.timeout_seconds = job_request.timeout_seconds
        self.job_metadata = job_request.metadata
        self.status = job_request.status.value
        self.progress = job_request.progress
        self.assigned_model = job_request.assigned_model
        self.created_at = job_request.created_at
        self.started_at = job_request.started_at
        self.completed_at = job_request.completed_at
        self.result = job_request.result
        self.error = job_request.error
        self.retry_count = job_request.retry_count
        self.max_retries = job_request.max_retries
        self.last_retry_at = job_request.last_retry_at
        self.retry_reason = job_request.retry_reason
