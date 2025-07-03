"""FastAPI dependencies for dependency injection"""

import logging
from typing import Optional

from core.config import get_settings
from fastapi import Depends, Header, HTTPException, Request

logger = logging.getLogger(__name__)

async def get_current_settings():
    """Get current application settings"""
    return get_settings()

async def get_scheduler(request: Request):
    """Get the model scheduler instance from app state"""
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(
            status_code=503,
            detail="Model scheduler is not available. Please try again later."
        )
    return scheduler

async def verify_api_key(
    authorization: Optional[str] = Header(None),
    settings = Depends(get_current_settings)
):
    """Verify API key if authentication is enabled"""
    if not settings.security.api_key_required:
        return True
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing"
        )
    
    # Basic API key validation (you can enhance this)
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Use 'Bearer <token>'"
        )
    
    # For now, just check if token is present
    # You can implement proper JWT or API key validation here
    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    
    return True

async def check_rate_limit(
    request_ip: str = None,
    settings = Depends(get_current_settings)
):
    """Check rate limiting (basic implementation)"""
    # This is a placeholder for rate limiting
    # In production, you'd use Redis or similar for distributed rate limiting
    return True

class CommonQueryParams:
    """Common query parameters for API endpoints"""
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        order: str = "asc"
    ):
        self.skip = skip
        self.limit = min(limit, 1000)  # Cap at 1000
        self.sort_by = sort_by
        self.order = order
