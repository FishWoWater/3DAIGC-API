# 3DAIGC-API: Scalable 3D Generative AI Backend


[中文 README](README_zh.md)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/FishWoWater/3DAIGC-API.svg)](https://github.com/FishWoWater/3DAIGC-API/stargazers)
[![Code Size](https://img.shields.io/github/repo-size/FishWoWater/3DAIGC-API.svg)](https://github.com/FishWoWater/3DAIGC-API)

A FastAPI backend server framework for 3D generative AI models, with all of them ported to API-ready inference services, powered with GPU resource management and VRAM-aware scheduling.
> This project is still under active development and may contain breaking changes.

## 🏗️ System Architecture
The system provides a unified API gateway for multiple 3D AI models with automatic resource management:

```
          ┌─────────────────┐        ┌────────────────┐
          │   Client Apps   │        │  Web FrontEnd  │
          └─────────┬───────┘        └────────┬───────┘
                    │                         │
                    └─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    FastAPI Gateway      │
                    │   (Main Entry Point)    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼─────────┐ ┌─────▼─────┐ ┌─────────▼────────┐
    │       Router      │ │ Auth/Rate │ │   Job Scheduler  │
    │    & Validator    │ │ Limit(TBD)│ │    & Queue       │
    └─────────┬─────────┘ └───────────┘ └─────────┬────────┘
              │                                   │
              └───────────────┬───────────────────┘
                              │
                 ┌────────────▼────────────┐
                 │ Multiprocess Scheduler  |
                 │    (GPU Scheduling)     │
                 └────────────┬────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐   ┌────────▼────────┐
│  GPU Worker   │    │   GPU Worker    │   │   GPU Worker    │
│   (VRAM: 8GB) │    │  (VRAM: 24GB)   │   │  (VRAM: 4GB)    │
│     TRELLIS   │    │  Hunyuan3D-2.1  │   │ UniRig/PartField│
│     MeshGen   |    │      MeshGen    │   | AutoRig/MeshSeg │
└───────────────┘    └─────────────────┘   └─────────────────┘
```

### Key Features
- **VRAM-Aware Scheduling**: Intelligent GPU resource allocation based on model requirements
- **Dynamic Model Loading**: Models are loaded/unloaded on-demand to optimize memory usage
- **Multi-Model Support**: Unified interface for various 3D AI models
- **Async Processing**: Non-blocking request handling with job queue management
- **Format Flexibility**: Support for multiple input/output formats (GLB, OBJ, FBX)
- **RESTful API**: Clean, well-documented REST endpoints with OpenAPI specification

## 🤖 Supported Models & Features
The VRAM requirement is from the pytest results, tested on a single 4090 GPU.
### Text/Image to 3D Mesh Generation
| Model | Input | Output | VRAM | Features |
|-------|-------|--------|------|----------|
| **[TRELLIS](https://github.com/FishWoWater/TRELLIS)** | Text/Image | Textured Mesh | 12GB | Medium-quality, geometry & texture |
| **[Hunyuan3D-2.0mini](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)** | Image | Raw Mesh | 5GB | Very fast, medium quality, geometry only |
| **[Hunyuan3D-2.0mini](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)** | Image | Textured Mesh | 14GB | Medium-quality, geometry & texture |
| **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** | Image | Raw Mesh | 8GB | Fast, medium quality, geometry only |
| **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** | Image | Textured Mesh | 19GB | High-quality, geometry & texture |
| **[PartPacker](https://github.com/NVlabs/PartPacker)** | Image | Raw Mesh | 10GB | Part-Level, geometry only |

### Automatic Rigging 
| Model | Input | Output | VRAM | Features |
|-------|-------|--------|------|----------|
| **[UniRig](https://github.com/VAST-AI-Research/UniRig)** | Mesh | Rigged Mesh | 9GB | Automatic skeleton generation |

### Mesh Segmentation 
| Model | Input | Output | VRAM | Features |
|-------|-------|--------|------|----------|
| **[PartField](https://github.com/nv-tlabs/PartField)** | Mesh | Segmented Mesh | 4GB | Semantic part segmentation |

### Part Completion
| Model | Input | Output | VRAM | Features |
|-------|-------|--------|------|----------|
| **[HoloPart](https://github.com/VAST-AI-Research/HoloPart)** | Partial Mesh | Complete Mesh | 10GB | Part completion |

### Texture Generation (Mesh Painting)
| Model | Input | Output | VRAM | Features |
|-------|-------|--------|------|----------|
| **[TRELLIS Paint](https://github.com/FishWoWater/TRELLIS)** | Text/Image + Mesh | Textured Mesh | 8GB/4GB | Text/image-guided painting |
| **[Hunyuan3D-2.0 Paint](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)** | Mesh + Image   | Textured Mesh | 11GB | Medium-quality texture synthesis |
| **[Hunyuan3D-2.1 Paint](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** | Mesh + Image   | Textured Mesh | 12GB | High-quality texture synthesis, PBR |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (each model has its own VRAM requirement)
- Linux (tested on Ubuntu 20.04 and CentOS8)

### Installation
1. **Clone the repository**:
```bash
# clone this repo recursively 
# notice that the support of a number of models is based on MY FORK TO add API-READY interfaces
git clone --recurse-submodules https://github.com/FishWoWater/3DAIGC-API.git
cd 3DAIGC-API
```

2. **Run the installation script**:
```bash
# on linux
chmod +x install.sh
./scripts/install.sh
# on windows 
.\scripts\install.bat 
```
The installation script will:
- Set up the TRELLIS environment as base.
- Install all model dependencies (PartField, Hunyuan3D, HoloPart, UniRig, PartPacker).
- Install FastAPI and backend dependencies.

3. **Download pre-trained models(optional, or download automatically)**:
```bash
# on linux 
chmod +x download_models.sh
# download specific model(s)
./scripts/download_models.sh -m partfield,trellis
# Verify all existing models
./scripts/download_models.sh -v
# Show help
./scripts/download_models.sh -h
# List available models
./scripts/download_models.sh --list

# on windows, simiarly 
.\scripts\download_models.bat
```

### Running the Server
```bash
# on linux 
chmod +x scripts/run_server.sh
# development mode (auto-reload)
P3D_RELOAD=true ./scripts/run_server.sh
# production mode, (notice that you also need to change the .yaml specification)
P3D_RELOAD=false ./scripts/run_server.sh
# custom configuration
P3D_HOST=0.0.0.0 P3D_PORT=7842  ./scripts/run_server.sh

# or on windows 
.\scripts\run_server.bat 
```
The server will be available at `http://localhost:7842` by default.

### API Documentation

Once the server is running, visit:
- **Interactive API docs**: `http://localhost:7842/docs`
- **ReDoc documentation**: `http://localhost:7842/redoc`

## 📖 Usage Examples
### Basics
```bash 
# check system status 
curl -X GET "http://localhost:7842/api/v1/system/status/"
# check available features 
curl -X GET "http://localhost:7842/api/v1/system/features"
# check available models 
curl -X GET "http://localhost:7842/api/v1/system/models"
```
<details>
<summary>Example Response Querying the Features</summary>

{
  "features":[
    {"name":"text_to_textured_mesh",
      "model_count":1,
      "models":["trellis_text_to_textured_mesh"]
    },
    {"name":"text_mesh_painting",
    "model_count":1,
    "models":["trellis_text_mesh_painting"]
    },
    {"name":"image_to_raw_mesh",
    "model_count":2,
    "models":["hunyuan3d_image_to_raw_mesh","partpacker_image_to_raw_mesh"]
    },
    {"name":"image_to_textured_mesh",
    "model_count":2,
    "models":["trellis_image_to_textured_mesh","hunyuan3d_image_to_textured_mesh"]
    },
    {"name":"image_mesh_painting",
    "model_count":2,
    "models":["trellis_image_mesh_painting","hunyuan3d_image_mesh_painting"]
    },
    {"name":"mesh_segmentation",
    "model_count":1,
    "models":["partfield_mesh_segmentation"]
    },
    {"name":"auto_rig",
    "model_count":1,
    "models":["unirig_auto_rig"]
    },
    {"name":"part_completion",
    "model_count":1,
    "models":["holopart_part_completion"]
    }
  ],
  "total_features":8
}
</details>

### Text to 3D Mesh
```bash
curl -X POST "http://localhost:7842/api/v1/mesh-generation/text-to-textured-mesh" \
  -H "Content-Type: application/json" \
  -d '{
    "text_prompt": "A cute robot cat",
    "output_format": "glb",
    "model_preference": "trellis_text_to_textured_mesh"
  }'
```

### Image to 3D Mesh
```bash
curl -X POST "http://localhost:7842/api/v1/mesh-generation/image-to-textured-mesh" \
  -F "image_file=@assets/example_image/typical_humanoid_mech.png" \
  -F "output_format=glb"
```

### Mesh Segmentation
```bash
curl -X POST "http://localhost:7842/api/v1/mesh-segmentation/segment-mesh" \
  -F "mesh_file=@assets/example_mesh/typical_creature_dragon.obj" \
  -F "model_preference=partfield_mesh_segmentation"
```

### Auto-Rigging
```bash
curl -X POST "http://localhost:7842/api/v1/auto-rigging/generate-rig" \
  -F "rig_mode=full" \
  -F "mesh_file=@assets/example_autorig/giraffe.glb" \
  -F "output_format=fbx"
```

## 🧪 Testing

### Adapter Tests
```bash
# test all the adapters 
python tests/run_adapter_tests.py 
# test some specific adapter e.g. testing trellis
PYTHONPATH=. pytest tests/test_adapters/test_trellis_adapter.py -v -s -r s
```

### Integration Tests
```bash
# submit a job, wait until finished, next job
python tests/run_test_client.py --server-url http://localhost:7842 \
  --timeout 600 --poll-interval 25 --output-dir test_results.sequential --sequential 

# submit all the jobs at once, then monitor all of them (default behavoir)
# timeout larger to cover all the jobs
python tests/run_test_client.py --server-url http://localhost:7842 \
  --timeout 3600 --poll-interval 30 --output-dir test_results.concurrent 
```

### Others
```bash
# Test basic api endpoints
pytest tests/test_basic_endpoints.py -v -s -r s
# Run the on-demand multiprocesing scheduler 
python tests/test_on_demand_scheduler.py 
```

## ⚙️ Configuration
* [system configuration](./config/system.yaml)
* [model configuration](./config/models.yaml)
* [logging configuration](./config/logging.yaml)



## 🔧 Development
### Project Structure
```
3DAIGC-API/
├── api/                   # FastAPI application
│   ├── main.py            # Entry point
│   └── routers/           # API endpoints
├── adapters/              # Model adapters
├── core/                  # Core framework
│   ├── scheduler/         # GPU scheduling
│   └── models/            # Model abstractions
├── config/                # Configuration files
├── tests/                 # Test suites
├── thirdparty/            # Third-party models
└── utils/                 # Utilities
```

### Adding New Models
1. Create an adapter in `adapters/` following the base interface
2. Register the model in `config/models.yaml` and model factory `core/scheduler/model_factory.py`
3. Add adapter test and/or integration tests in `tests`

## Notes 
1. Current system **ONLY SUPPORTS A SINGLE UVICORN WORKER**. Multiple workers hold multiple scheduler instances and break everything. But since AI inference is GPU-intensive and the worker is used to address web requests, so that **won't be the bottleneck** in small-scale use.
2. Frequently loading/unloading models is very slow (as can be observed in the test client). Better to enable ONLY required models and always keep them in the VRAM in practice.
3. A lot of the code is written by vibe coding (Cursor + Claude4), Claude4 is a good software engineer, and I have learnt a lot from him/her in system design. Have a look at [vibe coding prompt](./docs/vibe_coding_prompt.md) and [vibe coding READMEs](./docs/vibe_coding/) if interested.

## 🛣️ TODO
### Short-Term 
- [ ] Better orgnaize (cleanup) the output directory of current API service
- [ ] Support multiview images as the condition in mesh generation models
- [ ] Expose and support more parameters (e.g. decimation ratio in mesh generation)

### Long-Term 
- [ ] Windows one-click installer
- [ ] Job queue and scheduler switches to redis, may add rate limit
- [ ] Based on this collection of 3D API, replicate/implement a similar 3D studio like Tripo/Hunyuan, where the frontend and the backend can BOTH be deployed easily on personal PCs

## 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
Notice that each algorithm **HAS ITS OWN LICENSE**, please check them carefully if needed.

## 🌟 Acknowledgment
Special thanks to the authors and contributors of all integrated models and the open-source community :)