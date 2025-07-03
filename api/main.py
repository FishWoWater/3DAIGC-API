"""FastAPI application main entry point"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from core.config import create_directories, get_settings, setup_logging
from core.scheduler import GPUMonitor
from core.scheduler.scheduler_factory import (
    create_development_scheduler,
    create_production_scheduler,
)
from core.utils.exceptions import BaseAPIException

from .routers import auto_rigging, mesh_generation, mesh_segmentation, system

logger = logging.getLogger(__name__)

# Global variables for shared resources
scheduler = None


# Configure CORS
def configure_cors(app: FastAPI, settings):
    """Configure CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


# Configure security middleware
def configure_security(app: FastAPI, settings):
    """Configure security middleware"""
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"],
            # allowed_hosts=["localhost", "127.0.0.1", settings.server.host]
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global scheduler

    # Startup
    logger.info("Starting 3D Generative Models Backend...")

    try:
        # Load configuration
        settings = get_settings()

        # Setup logging
        setup_logging(settings.logging)

        # Create necessary directories
        create_directories(settings.storage)

        # Initialize GPU monitor and scheduler
        gpu_monitor = GPUMonitor(memory_buffer=1024)  # Keep 1GB free

        # Use factory to create appropriate scheduler based on environment
        logger.info(f"Environment: {settings.environment}")
        if settings.environment == "production":
            logger.info("Creating production scheduler (multiprocess)")
            scheduler = create_production_scheduler(
                gpu_monitor=gpu_monitor,
                models_config=settings.models,
            )
        else:
            logger.info("Creating development scheduler (async)")
            scheduler = create_development_scheduler(
                gpu_monitor=gpu_monitor,
                models_config=settings.models,
            )

        # Note: The factory functions handle model registration automatically
        # when auto_register_models=True (which is the default)
        logger.info("Scheduler created with automatic model registration")

        # Start the scheduler
        await scheduler.start()

        # Store scheduler in app state for dependency injection
        app.state.scheduler = scheduler

        logger.info("Application startup completed successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

    finally:
        # Shutdown
        logger.info("Shutting down 3D Generative Models Backend...")

        # Cleanup resources
        if scheduler:
            await scheduler.stop()

        logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="3D Generative Models API",
    description="Scalable 3D AI model inference server with VRAM-aware scheduling",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS and security middleware
settings = get_settings()
configure_cors(app, settings)
configure_security(app, settings)


# Add middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()

    logger.info(f"Request: {request.method} {request.url}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} - "
        f"{request.method} {request.url} - "
        f"Time: {process_time:.3f}s"
    )

    return response


# Exception handlers
@app.exception_handler(BaseAPIException)
async def base_api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions"""
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.error_code or "API_ERROR",
            "message": exc.message,
            "detail": str(exc),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "INVALID_VALUE",
            "message": "Invalid input value",
            "detail": str(exc),
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "NOT_FOUND",
            "message": "Resource not found",
            "detail": str(exc.detail)
            if hasattr(exc, "detail")
            else "The requested resource was not found",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An internal server error occurred",
            "detail": "Please try again later or contact support",
        },
    )


# Include routers
app.include_router(system.router, prefix="/api/v1/system", tags=["System"])

app.include_router(mesh_generation.router, prefix="/api/v1", tags=["Mesh Generation"])


app.include_router(auto_rigging.router, prefix="/api/v1", tags=["Auto Rigging"])

app.include_router(
    mesh_segmentation.router, prefix="/api/v1", tags=["Mesh Segmentation"]
)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time(), "version": "1.0.0"}


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "3D Generative Models API",
        "version": "1.0.0",
        "description": "Scalable 3D AI model inference server",
        "docs_url": "/docs",
        "health_url": "/health",
    }
