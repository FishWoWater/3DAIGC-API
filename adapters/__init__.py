"""
Model adapters for integrating specific AI models into the framework.

This package contains adapters that bridge between our model specifications
and actual AI model implementations.
"""

from .holopart_adapter import HoloPartCompletionAdapter
from .hunyuan3d_adapter_v20 import (
    Hunyuan3DV20ImageMeshPaintingAdapter,
    Hunyuan3DV20ImageToRawMeshAdapter,
    Hunyuan3DV20ImageToTexturedMeshAdapter,
)
from .hunyuan3d_adapter_v21 import (
    Hunyuan3DV21ImageMeshPaintingAdapter,
    Hunyuan3DV21ImageToRawMeshAdapter,
    Hunyuan3DV21ImageToTexturedMeshAdapter,
)
from .partfield_adapter import PartFieldSegmentationAdapter
from .partpacker_adapter import PartPackerImageToRawMeshAdapter
from .trellis_adapter import (
    TrellisImageMeshPaintingAdapter,
    TrellisImageToTexturedMeshAdapter,
    TrellisTextMeshPaintingAdapter,
    TrellisTextToTexturedMeshAdapter,
)
from .unirig_adapter import UniRigAdapter

__all__ = [
    "HoloPartCompletionAdapter",
    "Hunyuan3DV20ImageMeshPaintingAdapter",
    "Hunyuan3DV20ImageToRawMeshAdapter",
    "Hunyuan3DV20ImageToTexturedMeshAdapter",
    "Hunyuan3DV21ImageMeshPaintingAdapter",
    "Hunyuan3DV21ImageToRawMeshAdapter",
    "Hunyuan3DV21ImageToTexturedMeshAdapter",
    "PartFieldSegmentationAdapter",
    "PartPackerImageToRawMeshAdapter",
    "TrellisImageMeshPaintingAdapter",
    "TrellisImageToTexturedMeshAdapter",
    "TrellisTextMeshPaintingAdapter",
    "TrellisTextToTexturedMeshAdapter",
    "UniRigAdapter",
]
