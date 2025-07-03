# Comprehensive 3D Generative Models Test Client

This directory contains a comprehensive test client that automatically tests all available 3D generative models and features using example assets. The test client submits jobs, monitors their progress, downloads results, and organizes outputs in a structured directory.

## Features Tested

The test client covers all major features and models available in the 3D generative API:

### 1. Text-to-Textured-Mesh
- **Models**: TRELLIS
- **Tests**: English and Chinese prompts with texture generation
- **Assets**: Text prompts from `assets/example_prompts.txt`

### 2. Text-Mesh-Painting  
- **Models**: TRELLIS
- **Tests**: Apply text-described textures to existing meshes
- **Assets**: OBJ mesh files from `assets/example_mesh/`

### 3. Image-to-Textured-Mesh
- **Models**: TRELLIS, Hunyuan3D v2.0, Hunyuan3D v2.1
- **Tests**: Generate textured meshes from input images
- **Assets**: PNG images from `assets/example_image/`

### 4. Image-Mesh-Painting
- **Models**: TRELLIS, Hunyuan3D v2.0, Hunyuan3D v2.1  
- **Tests**: Apply image-based textures to existing meshes
- **Assets**: Images + mesh combinations

### 5. Image-to-Raw-Mesh
- **Models**: Hunyuan3D v2.0, Hunyuan3D v2.1, PartPacker
- **Tests**: Generate geometry-only meshes from images
- **Assets**: Images from various example directories

### 6. Mesh Segmentation
- **Models**: PartField
- **Tests**: Segment meshes into semantic parts  
- **Assets**: GLB files from `assets/example_meshseg/`

### 7. Part Generation/Completion
- **Models**: HoloPart
- **Tests**: Complete partial or missing mesh parts
- **Assets**: GLB files from `assets/example_holopart/`

### 8. Auto-Rigging
- **Models**: UniRig
- **Tests**: Automatically add bone structures to character meshes
- **Assets**: GLB files from `assets/example_autorig/`

## Usage

### Quick Start

The easiest way to run the comprehensive tests is using the provided shell script:

```bash
# Run with default settings (server at localhost:7842)
./tests/run_comprehensive_tests.sh

# Run against a remote server
./tests/run_comprehensive_tests.sh --server-url http://10.0.0.1:8000

# Use custom timeout and output directory
./tests/run_comprehensive_tests.sh --timeout 1200 --output-dir my_results
```

### Direct Python Usage

You can also run the test client directly:

```bash
# Basic usage
python3 tests/comprehensive_model_test_client.py

# With custom parameters
python3 tests/comprehensive_model_test_client.py \
    --server-url http://localhost:7842 \
    --timeout 600 \
    --poll-interval 10 \
    --output-dir test_results
```

### Command Line Options

#### Shell Script Options (`run_comprehensive_tests.sh`)
- `-s, --server-url URL`: Server URL (default: http://localhost:7842)
- `-t, --timeout SECONDS`: Timeout per job in seconds (default: 600)
- `-p, --poll-interval SEC`: Poll interval in seconds (default: 10)
- `-o, --output-dir DIR`: Output directory (default: test_results)
- `-h, --help`: Show help message

#### Python Script Options (`comprehensive_model_test_client.py`)
- `--server-url`: Server URL
- `--timeout`: Timeout per job in seconds
- `--poll-interval`: Job status polling interval
- `--output-dir`: Output directory for results

## Output Structure

The test client creates an organized directory structure for results:

```
test_results/
├── logs/
│   └── test_client.log                    # Detailed execution logs
├── text_to_textured_mesh/
│   ├── trellis_text_to_textured_mesh_english_result.glb
│   └── trellis_text_to_textured_mesh_chinese_result.glb
├── text_mesh_painting/
│   ├── trellis_text_mesh_painting_typical_creature_dragon_result.glb
│   └── trellis_text_mesh_painting_typical_humanoid_mech_result.glb
├── image_to_textured_mesh/
│   ├── trellis_image_to_textured_mesh_073_result.glb
│   ├── hunyuan3d_2_0_image_to_textured_mesh_073_result.glb
│   └── hunyuan3d_image_to_textured_mesh_073_result.glb
├── image_mesh_painting/
│   └── trellis_image_mesh_painting_073_typical_creature_dragon_result.glb
├── image_to_raw_mesh/
│   ├── hunyuan3d_2_0_image_to_raw_mesh_073_result.glb
│   ├── hunyuan3d_image_to_raw_mesh_073_result.glb
│   ├── partpacker_part_packing_barrel_result.glb
│   └── partpacker_part_packing_cactus_result.glb
├── mesh_segmentation/
│   ├── partfield_mesh_segmentation_00200996b8f34f55a2dd2f44d316d107_result.glb
│   └── partfield_mesh_segmentation_002e462c8bfa4267a9c9f038c7966f3b_result.glb
├── part_completion/
│   ├── holopart_part_completion_000_result.glb
│   └── holopart_part_completion_001_result.glb
├── auto_rigging/
│   ├── unirig_auto_rigging_bird_result.fbx
│   └── unirig_auto_rigging_giraffe_result.fbx
└── test_results.json                      # Detailed JSON results
```

## Test Results

### JSON Results File

The `test_results.json` file contains comprehensive information about all tests:

```json
{
  "summary": {
    "total_tests": 25,
    "successful": 20,
    "failed": 3,
    "skipped": 2,
    "start_time": 1703875200.0,
    "end_time": 1703878800.0
  },
  "tests": [
    {
      "test_name": "trellis_text_to_textured_mesh_english",
      "feature": "text_to_textured_mesh",
      "model_preference": "trellis_text_to_textured_mesh",
      "endpoint": "/mesh-generation/text-to-textured-mesh",
      "status": "success",
      "job_id": "job_123456",
      "start_time": 1703875200.0,
      "end_time": 1703875800.0,
      "duration": 600.0,
      "downloaded_file": "test_results/text_to_textured_mesh/trellis_text_to_textured_mesh_english_result.glb",
      "job_result": { /* Full job status response */ }
    }
    // ... more test results
  ]
}
```

### Status Types

Each test can have one of the following statuses:

- **success**: Test completed successfully and result was downloaded
- **submission_failed**: Failed to submit job to API
- **timeout**: Job took longer than the specified timeout
- **job_failed**: Job was submitted but failed during execution
- **download_failed**: Job completed but result download failed
- **exception**: Unexpected error occurred during test

### Summary Report

The test client generates a human-readable summary report:

```
================================================================================
COMPREHENSIVE 3D GENERATIVE MODELS TEST REPORT
================================================================================

OVERALL RESULTS:
  Total Tests:    25
  Successful:     20 (80.0%)
  Failed:         3 (12.0%)
  Skipped:        2 (8.0%)
  Duration:       3600.0 seconds

RESULTS BY FEATURE:
  text_to_textured_mesh      2/ 2 (100.0%)
  text_mesh_painting         2/ 2 (100.0%)
  image_to_textured_mesh     4/ 4 (100.0%)
  image_mesh_painting        2/ 2 (100.0%)
  image_to_raw_mesh          4/ 6 ( 66.7%)
  mesh_segmentation          2/ 2 (100.0%)
  part_completion            2/ 2 (100.0%)
  auto_rigging               2/ 2 (100.0%)

DETAILED RESULTS:
  ✓ trellis_text_to_textured_mesh_english          success         120.5s
    Output: test_results/text_to_textured_mesh/trellis_text_to_textured_mesh_english_result.glb
  ✓ trellis_text_to_textured_mesh_chinese          success         115.2s
    Output: test_results/text_to_textured_mesh/trellis_text_to_textured_mesh_chinese_result.glb
  ✗ partpacker_part_packing_barrel                 timeout         600.0s
    Error: Job timed out
  ...

OUTPUT DIRECTORY: test_results
================================================================================
```

## Prerequisites

### Server Requirements

1. **3D Generative API Server**: The server must be running and accessible
2. **Model Availability**: Models should be loaded and ready for inference
3. **Asset Access**: Server must have access to the `assets/` directory

### Client Requirements

1. **Python 3.7+**: Required for the test client
2. **requests library**: Automatically installed if missing
3. **Network Access**: Ability to reach the server URL

### System Requirements

- **Disk Space**: Sufficient space for downloading generated 3D models (varies by model output)
- **Time**: Tests can take 30 minutes to several hours depending on:
  - Number of available models
  - Model inference speed
  - Server load
  - Network bandwidth

## Configuration

### Test Configuration

The test client automatically configures tests based on available assets:

- **Text Prompts**: Read from `assets/example_prompts.txt`
- **Images**: Automatically discovered from various asset directories
- **Meshes**: Automatically discovered from mesh asset directories
- **Model Selection**: Uses 1-2 assets per model to avoid overwhelming the server

### Asset Requirements

Ensure the following asset directories exist and contain files:

```
assets/
├── example_prompts.txt          # Text prompts (any encoding)
├── example_image/               # PNG/JPG images for image-based tests
├── example_mesh/                # OBJ files for mesh painting tests
├── example_meshseg/            # GLB files for segmentation tests  
├── example_holopart/           # GLB files for part completion tests
├── example_autorig/            # GLB files for auto-rigging tests
└── example_partpacker/         # PNG files for PartPacker tests
```

## Troubleshooting

### Common Issues

#### Server Connection Errors
```bash
# Check if server is running
curl http://localhost:7842/health

# Check available models
curl http://localhost:7842/api/v1/system/available-adapters
```

#### Missing Assets
```bash
# Verify asset directories exist
ls -la assets/
ls -la assets/example_*
```

#### Permission Errors
```bash
# Make scripts executable
chmod +x tests/run_comprehensive_tests.sh
chmod +x tests/comprehensive_model_test_client.py
```

#### Python Dependencies
```bash
# Install required packages manually
pip3 install requests
```

### Debugging

1. **Check Logs**: Detailed logs are written to `test_results/logs/test_client.log`
2. **Individual Tests**: Run single model tests by modifying the test configurations
3. **Server Status**: Check server logs for model loading and inference errors
4. **Resource Usage**: Monitor GPU/CPU usage during tests

### Timeout Issues

If jobs are timing out:

1. **Increase Timeout**: Use `--timeout 1200` for longer jobs
2. **Check Server Load**: Ensure server isn't overloaded
3. **Model Status**: Verify models are loaded and not in error state
4. **Resource Availability**: Check GPU memory availability

## Integration with CI/CD

The test client can be integrated into automated testing pipelines:

```bash
# Run tests and check exit code
./tests/run_comprehensive_tests.sh --output-dir ci_results
EXIT_CODE=$?

# Parse results for CI reporting
python3 -c "
import json, sys
with open('ci_results/test_results.json') as f:
    results = json.load(f)
    
failed = results['summary']['failed']
total = results['summary']['total_tests']

print(f'Comprehensive Tests: {total-failed}/{total} passed')
sys.exit(1 if failed > 0 else 0)
"
```

## Examples

### Basic Test Run

```bash
# Start server (in another terminal)
cd /path/to/3DAIGC-API
python api/main.py

# Run comprehensive tests
./tests/run_comprehensive_tests.sh
```

### Custom Configuration

```bash
# Test against production server with longer timeout
./tests/run_comprehensive_tests.sh \
    --server-url https://api.3dgen.example.com \
    --timeout 1800 \
    --output-dir production_test_results
```

### Selective Testing

To test only specific features, modify the `test_configs` in `comprehensive_model_test_client.py`:

```python
# Example: Test only TRELLIS models
configs = [config for config in configs if "trellis" in config["model_preference"]]
```

## Contributing

To add new models or features to the test suite:

1. **Add Test Configuration**: Update `setup_test_configurations()` method
2. **Add Assets**: Place example files in appropriate `assets/example_*` directories  
3. **Update Documentation**: Update this README with new feature information
4. **Test**: Run the comprehensive tests to verify new configurations work

## Performance

### Expected Durations

Typical test durations per model (on GPU server):

- **Text-to-textured-mesh**: 2-5 minutes
- **Image-to-textured-mesh**: 3-8 minutes  
- **Mesh painting**: 2-4 minutes
- **Image-to-raw-mesh**: 1-3 minutes
- **Mesh segmentation**: 1-2 minutes
- **Part generation**: 3-6 minutes
- **Auto-rigging**: 2-5 minutes

Total test suite duration: 30 minutes to 2+ hours depending on available models.

### Optimization

The test client is designed to:

- Run tests sequentially to avoid overwhelming the server
- Use minimal asset sets (1-2 files per model)
- Include progress logging and status monitoring
- Handle failures gracefully and continue testing
- Download results efficiently

For faster testing in development, consider:

- Testing fewer models at once
- Using smaller/simpler input assets
- Increasing poll intervals to reduce server load
- Running tests on a dedicated test server 