"""Validation utilities for input data"""

import base64
import io
import re
from typing import Optional

from PIL import Image

from .exceptions import ValidationError


def validate_text(text: str, max_length: int = 1000, min_length: int = 1) -> None:
    """Validate text input"""
    if not text or not text.strip():
        raise ValidationError("text", "Text cannot be empty")
    
    if len(text) < min_length:
        raise ValidationError("text", f"Text must be at least {min_length} characters")
    
    if len(text) > max_length:
        raise ValidationError("text", f"Text must be at most {max_length} characters")
    
    # Check for potentially harmful content (basic filter)
    harmful_patterns = [
        r'<script',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
    ]
    
    text_lower = text.lower()
    for pattern in harmful_patterns:
        if re.search(pattern, text_lower):
            raise ValidationError("text", "Text contains potentially harmful content")

def validate_image(image_b64: str, max_size_mb: int = 10) -> None:
    """Validate base64 encoded image"""
    if not image_b64:
        raise ValidationError("image", "Image data cannot be empty")
    
    try:
        # Remove data URL prefix if present
        if image_b64.startswith('data:image/'):
            header, image_b64 = image_b64.split(',', 1)
        
        # Decode base64
        image_data = base64.b64decode(image_b64)
        
        # Check file size
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError("image", f"Image size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")
        
        # Try to open with PIL to validate format
        image = Image.open(io.BytesIO(image_data))
        image.verify()  # Verify image integrity
        
        # Check image format
        allowed_formats = ['JPEG', 'PNG', 'WEBP']
        if image.format not in allowed_formats:
            raise ValidationError("image", f"Unsupported image format: {image.format}")
        
        # Check image dimensions
        width, height = image.size
        if width < 64 or height < 64:
            raise ValidationError("image", "Image dimensions too small (minimum 64x64)")
        
        if width > 4096 or height > 4096:
            raise ValidationError("image", "Image dimensions too large (maximum 4096x4096)")
            
    except Exception as decode_error:
        if "Invalid base64" in str(decode_error):
            raise ValidationError("image", "Invalid base64 encoding")
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError("image", f"Invalid image data: {str(e)}")

def validate_file_size(file_size: int, max_size_bytes: int) -> None:
    """Validate file size"""
    if file_size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise ValidationError("file_size", f"File size ({actual_mb:.1f}MB) exceeds limit ({max_mb:.1f}MB)")

def validate_model_preference(model_preference: str, available_models: list) -> None:
    """Validate model preference against available models"""
    if model_preference and model_preference not in available_models:
        raise ValidationError("model_preference", f"Model '{model_preference}' not available. Available models: {available_models}")

def validate_output_format(output_format: str, supported_formats: list) -> None:
    """Validate output format"""
    if output_format not in supported_formats:
        raise ValidationError("output_format", f"Format '{output_format}' not supported. Supported formats: {supported_formats}")

def validate_quality_setting(quality: str) -> None:
    """Validate quality setting"""
    allowed_qualities = ['low', 'medium', 'high']
    if quality not in allowed_qualities:
        raise ValidationError("quality", f"Quality must be one of: {allowed_qualities}")

def validate_seed(seed: Optional[int]) -> None:
    """Validate seed value"""
    if seed is not None:
        if not isinstance(seed, int):
            raise ValidationError("seed", "Seed must be an integer")
        if seed < 0 or seed > 2**32 - 1:
            raise ValidationError("seed", "Seed must be between 0 and 2^32-1")

def validate_texture_resolution(resolution: int) -> None:
    """Validate texture resolution"""
    allowed_resolutions = [512, 1024, 2048, 4096]
    if resolution not in allowed_resolutions:
        raise ValidationError("texture_resolution", f"Resolution must be one of: {allowed_resolutions}")

def validate_job_id(job_id: str) -> None:
    """Validate job ID format"""
    if not job_id:
        raise ValidationError("job_id", "Job ID cannot be empty")
    
    # UUID format validation
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, job_id.lower()):
        raise ValidationError("job_id", "Invalid job ID format")
