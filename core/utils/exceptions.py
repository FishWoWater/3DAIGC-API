"""Custom exceptions for the 3D Generative Models Backend"""

class BaseAPIException(Exception):
    """Base exception for API errors"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ModelNotFoundError(BaseAPIException):
    """Raised when a requested model is not found"""
    def __init__(self, model_id: str):
        super().__init__(f"Model '{model_id}' not found", "MODEL_NOT_FOUND")
        self.model_id = model_id

class InsufficientVRAMError(BaseAPIException):
    """Raised when there's insufficient VRAM to load a model"""
    def __init__(self, required_vram: int, available_vram: int):
        super().__init__(
            f"Insufficient VRAM: required {required_vram}MB, available {available_vram}MB",
            "INSUFFICIENT_VRAM"
        )
        self.required_vram = required_vram
        self.available_vram = available_vram

class JobTimeoutError(BaseAPIException):
    """Raised when a job times out"""
    def __init__(self, job_id: str, timeout: int):
        super().__init__(f"Job '{job_id}' timed out after {timeout} seconds", "JOB_TIMEOUT")
        self.job_id = job_id
        self.timeout = timeout

class ValidationError(BaseAPIException):
    """Raised when input validation fails"""
    def __init__(self, field: str, reason: str):
        super().__init__(f"Validation error for '{field}': {reason}", "VALIDATION_ERROR")
        self.field = field
        self.reason = reason

class ModelLoadError(BaseAPIException):
    """Raised when a model fails to load"""
    def __init__(self, model_id: str, reason: str):
        super().__init__(f"Failed to load model '{model_id}': {reason}", "MODEL_LOAD_ERROR")
        self.model_id = model_id
        self.reason = reason

class ProcessingError(BaseAPIException):
    """Raised when model processing fails"""
    def __init__(self, model_id: str, reason: str):
        super().__init__(f"Processing failed for model '{model_id}': {reason}", "PROCESSING_ERROR")
        self.model_id = model_id
        self.reason = reason

class FileUploadError(BaseAPIException):
    """Raised when file upload fails"""
    def __init__(self, filename: str, reason: str):
        super().__init__(f"File upload failed for '{filename}': {reason}", "FILE_UPLOAD_ERROR")
        self.filename = filename
        self.reason = reason

class ConfigurationError(BaseAPIException):
    """Raised when there's a configuration error"""
    def __init__(self, config_key: str, reason: str):
        super().__init__(f"Configuration error for '{config_key}': {reason}", "CONFIG_ERROR")
        self.config_key = config_key
        self.reason = reason
