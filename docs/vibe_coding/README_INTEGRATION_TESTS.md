# Integration Tests for 3D Generative Models Backend

This directory contains real integration tests for all adapters in the 3D generative models backend. Unlike the unit tests in `test_adapters/`, these tests run against a live server instance and perform actual end-to-end workflows.

## Test Files Overview

### 1. `test_trellis_adapter_integration.py`
**Features Tested:**
- Text-to-textured-mesh generation
- Text-based mesh painting
- Image-to-textured-mesh generation
- Image-based mesh painting

**API Endpoints:**
- `/api/v1/mesh-generation/text-to-textured-mesh`
- `/api/v1/mesh-generation/text-mesh-painting`
- `/api/v1/mesh-generation/image-to-textured-mesh`
- `/api/v1/mesh-generation/image-mesh-painting`

### 2. `test_hunyuan3d_adapter_integration.py`
**Features Tested:**
- Image-to-raw-mesh generation
- Image-to-textured-mesh generation
- Image-based mesh painting

**API Endpoints:**
- `/api/v1/mesh-generation/image-to-raw-mesh`
- `/api/v1/mesh-generation/image-to-textured-mesh`
- `/api/v1/mesh-generation/image-mesh-painting`

### 3. `test_partfield_adapter_integration.py`
**Features Tested:**
- Mesh segmentation into semantic parts

**API Endpoints:**
- `/api/v1/mesh-segmentation/segment-mesh`

### 4. `test_unirig_adapter_integration.py`
**Features Tested:**
- Skeleton generation
- Skin weights generation
- Full auto-rigging pipeline

**API Endpoints:**
- `/api/v1/auto-rigging/generate-rig`

### 5. `test_partpacker_adapter_integration.py`
**Features Tested:**
- Image-to-mesh generation (part packing approach)

**API Endpoints:**
- `/api/v1/mesh-generation/image-to-raw-mesh`

### 6. `test_holopart_adapter_integration.py`
**Features Tested:**
- Part completion from partial meshes

**API Endpoints:**
- `/api/v1/part-completion/complete` (or fallback endpoints)

## Prerequisites

### System Requirements
- Backend server running on `http://localhost:8000`
- Python 3.8+ with pytest installed
- GPU with sufficient VRAM:
  - TRELLIS: 6GB+ VRAM
  - Hunyuan3D: 8GB+ VRAM
  - PartField: 4GB+ VRAM
  - UniRig: 6GB+ VRAM
  - PartPacker: 4GB+ VRAM
  - HoloPart: 6GB+ VRAM

### Model Requirements
All corresponding models must be downloaded and available:
- TRELLIS models in `pretrained/TRELLIS/`
- Hunyuan3D models in `pretrained/tencent/Hunyuan3D-2.1/`
- PartField models in `pretrained/PartField/`
- UniRig models in `pretrained/UniRig/`
- PartPacker models in `pretrained/PartPacker/`
- HoloPart models in `pretrained/HoloPart/`

### Test Assets
Required test assets in the `assets/` directory:
- Example images: `assets/example_image/`
- Example meshes: `assets/example_mesh/`
- Auto-rigging examples: `assets/example_autorig/`
- Part completion examples: `assets/example_holopart/`
- PartPacker examples: `assets/example_partpacker/`

## Running Tests

### Run All Integration Tests
```bash
pytest tests/test_integration/ -m integration -v -s
```

### Run Specific Adapter Tests
```bash
# TRELLIS tests
pytest tests/test_integration/test_trellis_adapter_integration.py -v -s

# Hunyuan3D tests
pytest tests/test_integration/test_hunyuan3d_adapter_integration.py -v -s

# PartField tests
pytest tests/test_integration/test_partfield_adapter_integration.py -v -s

# UniRig tests
pytest tests/test_integration/test_unirig_adapter_integration.py -v -s

# PartPacker tests
pytest tests/test_integration/test_partpacker_adapter_integration.py -v -s

# HoloPart tests
pytest tests/test_integration/test_holopart_adapter_integration.py -v -s
```

### Run Specific Test Cases
```bash
# Test basic text-to-mesh generation
pytest tests/test_integration/test_trellis_adapter_integration.py::TestRealTextToTexturedMeshIntegration::test_text_to_textured_mesh -v -s

# Test image-to-mesh generation
pytest tests/test_integration/test_hunyuan3d_adapter_integration.py::TestRealImageToMeshIntegration::test_image_to_raw_mesh -v -s

# Test mesh segmentation
pytest tests/test_integration/test_partfield_adapter_integration.py::TestRealMeshSegmentationIntegration::test_mesh_segmentation_basic -v -s

# Test auto-rigging
pytest tests/test_integration/test_unirig_adapter_integration.py::TestRealAutoRiggingIntegration::test_skeleton_generation -v -s
```

### Run Only Fast Tests (Skip Slow Tests)
```bash
pytest tests/test_integration/ -m "integration and not slow" -v -s
```

### Run Only Slow Tests
```bash
pytest tests/test_integration/ -m "integration and slow" -v -s
```

## Test Categories

### Integration Tests (`@pytest.mark.integration`)
- Test complete end-to-end workflows
- Submit jobs via API
- Monitor job completion
- Validate outputs

### Slow Tests (`@pytest.mark.slow`)
- Tests that involve actual model inference
- May take several minutes to complete
- Generate real 3D content

### Fast Tests (no `@pytest.mark.slow`)
- API validation tests
- Job submission tests
- Status monitoring tests
- Usually complete in seconds

## Common Test Patterns

### Job Submission and Monitoring
All integration tests follow this pattern:
1. Submit job via POST request
2. Receive job ID
3. Poll job status until completion
4. Validate the generated output

### Health Checks
Each test suite includes:
- Server availability check
- Adapter availability check
- Automatic skip if prerequisites not met

### Output Validation
Tests validate:
- File existence and non-zero size
- Expected file formats
- Metadata and generation info
- Model-specific requirements

## Troubleshooting

### Common Issues

**Server Not Available**
```
Server not available: Connection refused
```
- Ensure backend server is running on port 8000
- Check server health: `curl http://localhost:8000/health`

**Model Not Available**
```
pytest.skip: TRELLIS text-to-textured-mesh adapter not available
```
- Download required models using `./download_models.sh`
- Check model paths in configuration files

**Insufficient GPU Memory**
```
Job failed with error: CUDA out of memory
```
- Close other GPU processes
- Reduce batch sizes or resolution in test requests
- Use a GPU with more VRAM

**Test Assets Missing**
```
AssertionError: Example image not found
```
- Ensure all test assets are present in `assets/` directory
- Check file paths match those expected by tests

### Performance Tips

**Reduce Test Time**
- Run only fast tests: `-m "integration and not slow"`
- Use lower resolutions in test requests
- Test fewer examples with parametrized tests

**Parallel Execution**
- Tests are designed to run sequentially due to GPU memory constraints
- Running multiple GPU tests in parallel may cause OOM errors

## Test Configuration

### Timeouts
- Standard job timeout: 300 seconds (5 minutes)
- Extended timeout for complex jobs: 600 seconds (10 minutes)
- Server health check timeout: 10 seconds

### Test Data
- Lower resolutions used for faster testing
- Reduced iteration counts where applicable
- Example assets chosen for quick processing

### Markers
- `@pytest.mark.integration`: All integration tests
- `@pytest.mark.slow`: Tests involving actual inference
- `@pytest.mark.timeout(N)`: Tests with custom timeouts

## Expected Outputs

### Successful Test Run
```
tests/test_integration/test_trellis_adapter_integration.py::TestRealTextToTexturedMeshIntegration::test_text_to_textured_mesh 
Job submitted successfully with ID: abc123
Job abc123 status: queued
Job abc123 status: processing
Job abc123 status: completed
Job completed successfully in 45.2 seconds
Generated mesh: /tmp/output_abc123.glb
Vertices: 2048, Faces: 4096
File size: 1245.6 KB
PASSED
```

### Test Skip (Expected)
```
tests/test_integration/test_hunyuan3d_adapter_integration.py::TestRealImageToMeshIntegration::test_image_to_raw_mesh 
SKIPPED (Server not available)
```

This comprehensive test suite ensures that all adapters work correctly through the complete API pipeline, providing confidence in the production deployment of the 3D generative models backend. 