"""
GPU memory tracking decorator for PyTorch-based tests.

This module provides a decorator to track peak GPU memory usage during test execution.
"""

import functools
import inspect
from typing import Any, Callable

import torch


def track_gpu_memory(func: Callable) -> Callable:
    """
    Decorator to track peak GPU memory usage during function execution.

    This decorator monitors GPU memory usage before, during, and after
    function execution, reporting the peak memory usage.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function with GPU memory tracking
    """

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            if not torch.cuda.is_available():
                # If CUDA is not available, just run the function normally
                return await func(*args, **kwargs)

            print(f"\n[GPU Memory Tracker] Starting {func.__name__}...")

            # Clear cache and record initial memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Reset peak memory tracker
            torch.cuda.reset_peak_memory_stats()

            try:
                # Execute the original function
                result = await func(*args, **kwargs)

                # Get peak and final memory usage
                peak_memory = torch.cuda.max_memory_allocated()
                final_memory = torch.cuda.memory_allocated()

                # Report memory usage
                initial_mb = initial_memory / (1024**2)
                peak_mb = peak_memory / (1024**2)
                final_mb = final_memory / (1024**2)

                print(f"\n[GPU Memory Tracker] {func.__name__} completed:")
                print(f"  Initial: {initial_mb:.1f}MB")
                print(f"  Peak:    {peak_mb:.1f}MB")
                print(f"  Final:   {final_mb:.1f}MB")
                print(f"  Delta:   +{peak_mb - initial_mb:.1f}MB")

                return result

            except Exception as e:
                print(
                    f"\n[GPU Memory Tracker] {func.__name__} failed with exception: {e}"
                )
                raise

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            if not torch.cuda.is_available():
                # If CUDA is not available, just run the function normally
                return func(*args, **kwargs)

            print(f"\n[GPU Memory Tracker] Starting {func.__name__}...")

            # Clear cache and record initial memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Reset peak memory tracker
            torch.cuda.reset_peak_memory_stats()

            try:
                # Execute the original function
                result = func(*args, **kwargs)

                # Get peak and final memory usage
                peak_memory = torch.cuda.max_memory_allocated()
                final_memory = torch.cuda.memory_allocated()

                # Report memory usage
                initial_mb = initial_memory / (1024**2)
                peak_mb = peak_memory / (1024**2)
                final_mb = final_memory / (1024**2)

                print(f"\n[GPU Memory Tracker] {func.__name__} completed:")
                print(f"  Initial: {initial_mb:.1f}MB")
                print(f"  Peak:    {peak_mb:.1f}MB")
                print(f"  Final:   {final_mb:.1f}MB")
                print(f"  Delta:   +{peak_mb - initial_mb:.1f}MB")

                return result

            except Exception as e:
                print(
                    f"\n[GPU Memory Tracker] {func.__name__} failed with exception: {e}"
                )
                raise

        return sync_wrapper
