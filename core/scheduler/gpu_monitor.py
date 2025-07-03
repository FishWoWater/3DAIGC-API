import asyncio
import logging
from typing import Dict, List, Optional

try:
    import GPUtil
except ImportError:
    GPUtil = None

import torch

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU resources and usage"""

    def __init__(self, memory_buffer: int = 300, tracking_mode: bool = True):
        self.memory_buffer = memory_buffer  # MB to keep free
        self.tracking_mode = (
            tracking_mode  # Use allocation tracking instead of real-time queries
        )
        self.gpus = []
        self._has_gputil = GPUtil is not None

        # For tracking mode: maintain allocated VRAM per GPU
        self.allocated_vram_per_gpu: Dict[int, int] = {}  # gpu_id -> allocated_vram_mb
        self.gpu_total_vram: Dict[int, int] = {}  # gpu_id -> total_vram_mb

        self._update_gpu_list()
        self._initialize_tracking()

    def _update_gpu_list(self):
        """Update list of available GPUs"""
        if torch.cuda.is_available() and self._has_gputil and GPUtil:
            try:
                self.gpus = GPUtil.getGPUs()
            except Exception as e:
                logger.warning(f"Failed to get GPU list with GPUtil: {e}")
                self.gpus = []
        elif torch.cuda.is_available():
            # Fallback to torch-only implementation
            gpu_count = torch.cuda.device_count()
            self.gpus = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                self.gpus.append(
                    {
                        "id": i,
                        "name": props.name,
                        "memory_total": props.total_memory
                        // (1024 * 1024),  # Convert to MB
                    }
                )
        else:
            self.gpus = []
            logger.warning("CUDA not available, running in CPU mode")

    def _initialize_tracking(self):
        """Initialize VRAM tracking for all GPUs"""
        if not self.tracking_mode:
            return

        self.allocated_vram_per_gpu.clear()
        self.gpu_total_vram.clear()

        if torch.cuda.is_available():
            for gpu_info in self.gpus:
                if isinstance(gpu_info, dict):
                    gpu_id = gpu_info["id"]
                    total_vram = gpu_info["memory_total"]
                else:
                    gpu_id = gpu_info.id
                    total_vram = gpu_info.memoryTotal

                if gpu_id is not None:
                    self.allocated_vram_per_gpu[gpu_id] = 0
                    self.gpu_total_vram[gpu_id] = total_vram

        logger.info(
            f"Initialized VRAM tracking for {len(self.gpu_total_vram)} GPUs: {self.gpu_total_vram}"
        )

    def allocate_vram(self, gpu_id: int, vram_mb: int) -> bool:
        """
        Allocate VRAM on a specific GPU (tracking mode only).

        Args:
            gpu_id: GPU ID to allocate VRAM on
            vram_mb: Amount of VRAM to allocate in MB

        Returns:
            True if allocation successful, False if insufficient VRAM
        """
        if not self.tracking_mode:
            logger.warning("allocate_vram called but tracking mode is disabled")
            return True

        if gpu_id not in self.allocated_vram_per_gpu:
            logger.error(f"GPU {gpu_id} not found in tracking")
            return False

        available_vram = self.get_gpu_available_vram(gpu_id)
        if available_vram >= vram_mb:
            self.allocated_vram_per_gpu[gpu_id] += vram_mb
            logger.info(
                f"Allocated {vram_mb}MB VRAM on GPU {gpu_id}, total allocated: {self.allocated_vram_per_gpu[gpu_id]}MB"
            )
            return True
        else:
            logger.warning(
                f"Insufficient VRAM on GPU {gpu_id}: need {vram_mb}MB, available {available_vram}MB"
            )
            return False

    def deallocate_vram(self, gpu_id: int, vram_mb: int):
        """
        Deallocate VRAM from a specific GPU (tracking mode only).

        Args:
            gpu_id: GPU ID to deallocate VRAM from
            vram_mb: Amount of VRAM to deallocate in MB
        """
        if not self.tracking_mode:
            logger.warning("deallocate_vram called but tracking mode is disabled")
            return

        if gpu_id not in self.allocated_vram_per_gpu:
            logger.error(f"GPU {gpu_id} not found in tracking")
            return

        self.allocated_vram_per_gpu[gpu_id] = max(
            0, self.allocated_vram_per_gpu[gpu_id] - vram_mb
        )
        logger.info(
            f"Deallocated {vram_mb}MB VRAM from GPU {gpu_id}, total allocated: {self.allocated_vram_per_gpu[gpu_id]}MB"
        )

    def get_gpu_available_vram(self, gpu_id: int) -> int:
        """
        Get available VRAM for a specific GPU.

        Args:
            gpu_id: GPU ID to check

        Returns:
            Available VRAM in MB (accounting for buffer and allocated VRAM in tracking mode)
        """
        if not torch.cuda.is_available():
            return 0

        if self.tracking_mode:
            if gpu_id not in self.gpu_total_vram:
                return 0
            total_vram = self.gpu_total_vram[gpu_id]
            allocated_vram = self.allocated_vram_per_gpu.get(gpu_id, 0)
            available_vram = total_vram - allocated_vram - self.memory_buffer
            return max(0, available_vram)
        else:
            # Real-time query mode
            gpu_status = self.get_gpu_status()
            for gpu_info in gpu_status:
                if gpu_info["id"] == gpu_id:
                    return max(0, gpu_info["memory_free"] - self.memory_buffer)
            return 0

    def get_gpu_status(self) -> List[Dict]:
        """Get current status of all GPUs"""
        self._update_gpu_list()
        status = []

        if self._has_gputil and torch.cuda.is_available():
            # Use GPUtil for detailed stats
            for gpu in self.gpus:
                gpu_status = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "memory_utilization": gpu.memoryUtil,
                    "gpu_utilization": gpu.load,
                    "temperature": gpu.temperature,
                }

                # Add tracking info if in tracking mode
                if self.tracking_mode and gpu.id in self.allocated_vram_per_gpu:
                    gpu_status["allocated_vram_tracked"] = self.allocated_vram_per_gpu[
                        gpu.id
                    ]
                    gpu_status["available_vram_tracked"] = self.get_gpu_available_vram(
                        gpu.id
                    )

                status.append(gpu_status)
        elif torch.cuda.is_available():
            # Use torch for basic memory info
            for gpu in self.gpus:
                gpu_id = gpu["id"]
                memory_info = torch.cuda.mem_get_info(gpu_id)
                memory_free = memory_info[0] // (1024 * 1024)  # Convert to MB
                memory_total = memory_info[1] // (1024 * 1024)
                memory_used = memory_total - memory_free

                gpu_status = {
                    "id": gpu_id,
                    "name": gpu["name"],
                    "memory_total": memory_total,
                    "memory_used": memory_used,
                    "memory_free": memory_free,
                    "memory_utilization": memory_used / memory_total
                    if memory_total > 0
                    else 0,
                    "gpu_utilization": 0.0,  # Not available without GPUtil
                    "temperature": 0,  # Not available without GPUtil
                }

                # Add tracking info if in tracking mode
                if self.tracking_mode and gpu_id in self.allocated_vram_per_gpu:
                    gpu_status["allocated_vram_tracked"] = self.allocated_vram_per_gpu[
                        gpu_id
                    ]
                    gpu_status["available_vram_tracked"] = self.get_gpu_available_vram(
                        gpu_id
                    )

                status.append(gpu_status)

        return status

    def find_best_gpu(self, required_memory: int) -> Optional[int]:
        """Find GPU with enough free memory"""
        self._update_gpu_list()

        if not torch.cuda.is_available():
            return None

        best_gpu = None
        max_free_memory = 0

        # Check each GPU for available VRAM
        for gpu_info in self.gpus:
            if isinstance(gpu_info, dict):
                gpu_id = gpu_info["id"]
            else:
                gpu_id = gpu_info.id

            if gpu_id is not None:
                available_memory = self.get_gpu_available_vram(gpu_id)

                if (
                    available_memory >= required_memory
                    and available_memory > max_free_memory
                ):
                    best_gpu = gpu_id
                    max_free_memory = available_memory

        return best_gpu

    def get_gpu_memory_usage(self, gpu_id: int) -> Dict:
        """Get memory usage for specific GPU"""
        if not torch.cuda.is_available():
            raise ValueError("CUDA not available")

        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU {gpu_id} not found")

        gpu_status = self.get_gpu_status()
        for gpu_info in gpu_status:
            if gpu_info["id"] == gpu_id:
                result = {
                    "total": gpu_info["memory_total"],
                    "used": gpu_info["memory_used"],
                    "free": gpu_info["memory_free"],
                    "utilization": gpu_info["memory_utilization"],
                }

                # Add tracking info if available
                if "allocated_vram_tracked" in gpu_info:
                    result["allocated_tracked"] = gpu_info["allocated_vram_tracked"]
                    result["available_tracked"] = gpu_info["available_vram_tracked"]

                return result

        raise ValueError(f"GPU {gpu_id} not found in status")

    async def monitor_continuously(self, interval: int = 5):
        """Continuously monitor GPU status"""
        while True:
            try:
                status = self.get_gpu_status()
                logger.debug(f"GPU Status: {status}")

                # Check for overheating or high utilization
                for gpu_info in status:
                    if gpu_info.get("temperature", 0) > 85:
                        logger.warning(
                            f"GPU {gpu_info['id']} temperature high: {gpu_info['temperature']}°C"
                        )

                    if gpu_info.get("gpu_utilization", 0) > 0.95:
                        logger.warning(
                            f"GPU {gpu_info['id']} utilization high: {gpu_info['gpu_utilization']:.1%}"
                        )

            except Exception as e:
                logger.error(f"Error monitoring GPUs: {str(e)}")

            await asyncio.sleep(interval)

    def get_total_vram(self) -> int:
        """Get total VRAM across all GPUs in MB"""
        if self.tracking_mode:
            return sum(self.gpu_total_vram.values())
        else:
            status = self.get_gpu_status()
            return sum(gpu["memory_total"] for gpu in status)

    def get_available_vram(self) -> int:
        """
        Get available VRAM across all GPUs in MB.

        Note: This method sums across all GPUs, but models typically only use one GPU.
        Consider using get_gpu_available_vram() for per-GPU allocation decisions.
        """
        if self.tracking_mode:
            return sum(
                self.get_gpu_available_vram(gpu_id)
                for gpu_id in self.gpu_total_vram.keys()
            )
        else:
            status = self.get_gpu_status()
            return sum(
                max(0, gpu["memory_free"] - self.memory_buffer) for gpu in status
            )
