 # 3DAIGC-API: 3D生成AI后端

[English README](README.md)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/FishWoWater/3DAIGC-API.svg)](https://github.com/FishWoWater/3DAIGC-API/stargazers)
[![Code Size](https://img.shields.io/github/languages/code-size/FishWoWater/3DAIGC-API.svg)](https://github.com/FishWoWater/3DAIGC-API)

一个基于 FastAPI 的 3D 生成AI模型后端服务框架，将所有模型移植为可用于 API 的推理服务，具备 GPU 资源管理和 VRAM 感知调度功能。

## 🏗️ 系统架构
系统为多个 3D AI 模型提供统一的 API 网关，具有自动资源管理功能：

```
          ┌─────────────────┐        ┌────────────────┐
          │   客户端应用    │        │   Web前端      │
          └─────────┬───────┘        └────────┬───────┘
                    │                         │
                    └─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    FastAPI 网关         │
                    │   (主要入口点)          │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼─────────┐ ┌─────▼─────┐ ┌─────────▼────────┐
    │       路由        │ │ 认证/限流 │ │   作业调度器     │
    │    & 验证器       │ │ (待定)    │ │    & 队列        │
    └─────────┬─────────┘ └───────────┘ └─────────┬────────┘
              │                                   │
              └───────────────┬───────────────────┘
                              │
                 ┌────────────▼────────────┐
                 │ 多进程调度器             |
                 │    (GPU 调度)           │
                 └────────────┬────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐   ┌────────▼────────┐
│  GPU Worker   │    │   GPU Worker    │   │   GPU Worker    │
│   (显存: 8GB) │    │  (显存: 24GB)   │   │  (显存: 4GB)    │
│     TRELLIS   │    │  Hunyuan3D-2.1  │   │ UniRig/PartField│
│     MeshGen   |    │      MeshGen    │   | AutoRig/MeshSeg │
└───────────────┘    └─────────────────┘   └─────────────────┘
```

### 核心特性
- **显存感知调度**: 基于模型需求的智能 GPU 资源分配
- **动态模型加载**: 按需加载/卸载模型以优化内存使用
- **多模型支持**: 为各种 3D AI 模型提供统一接口
- **异步处理**: 具有作业队列管理的非阻塞请求处理
- **格式灵活性**: 支持多种输入/输出格式 (GLB, OBJ, FBX)
- **RESTful API**: 清晰、文档完整的 REST 端点，具有 OpenAPI 规范

## 🤖 支持的模型和特性
显存需求来自 pytest 结果，在单个 4090 GPU 上测试。
### 文本/图像到3D网格生成
| 模型 | 输入 | 输出 | 显存 | 特性 |
|-------|-------|--------|------|----------|
| **[TRELLIS](https://github.com/FishWoWater/TRELLIS)** | 文本/图像 | 纹理网格 | 12GB | 中等质量，几何和纹理 |
| **[Hunyuan3D-2.0mini](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)** | 图像 | 原始网格 | 5GB | 非常快，中等质量，仅几何 |
| **[Hunyuan3D-2.0mini](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)** | 图像 | 纹理网格 | 14GB | 中等质量，几何和纹理 |
| **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** | 图像 | 原始网格 | 8GB | 快速，中等质量，仅几何 |
| **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** | 图像 | 纹理网格 | 19GB | 高质量，几何和纹理 |
| **[PartPacker](https://github.com/NVlabs/PartPacker)** | 图像 | 原始网格 | 10GB | 部件级，仅几何 |

### 自动绑定
| 模型 | 输入 | 输出 | 显存 | 特性 |
|-------|-------|--------|------|----------|
| **[UniRig](https://github.com/VAST-AI-Research/UniRig)** | 网格 | 绑定网格 | 9GB | 自动骨架生成 |

### 网格分割
| 模型 | 输入 | 输出 | 显存 | 特性 |
|-------|-------|--------|------|----------|
| **[PartField](https://github.com/nv-tlabs/PartField)** | 网格 | 分割网格 | 4GB | 语义部件分割 |

### 部件补全
| 模型 | 输入 | 输出 | 显存 | 特性 |
|-------|-------|--------|------|----------|
| **[HoloPart](https://github.com/VAST-AI-Research/HoloPart)** | 部分网格 | 完整网格 | 10GB | 部件补全 |

### 纹理生成 (网格绘制)
| 模型 | 输入 | 输出 | 显存 | 特性 |
|-------|-------|--------|------|----------|
| **[TRELLIS Paint](https://github.com/FishWoWater/TRELLIS)** | 网格 + 文本/图像 | 纹理网格 | 12GB | 文本/图像引导绘制 |
| **[Hunyuan3D-2.0 Paint](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)** | 网格 + 图像   | 纹理网格 | 11GB | 中等质量纹理合成 |
| **[Hunyuan3D-2.1 Paint](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** | 网格 + 图像   | 纹理网格 | 12GB | 高质量纹理合成，PBR |

## 🚀 快速开始

### 前置要求
- Python 3.10+
- CUDA 兼容的 GPU (每个模型有其自己的显存要求)
- Linux (在 Ubuntu 20.04 和 CentOS8 上测试)

### 安装
1. **克隆仓库**:
```bash
# 递归克隆此仓库
# 注意，多个模型的支持基于我的分支来添加 API-READY 服务
git clone --recurse-submodules https://github.com/FishWoWater/3DAIGC-API.git

cd 3DAIGC-API
```

2. **运行安装脚本**:
```bash
# Linux 系统
chmod +x install.sh
./scripts/install.sh

# Windows 系统
.\scripts\install.bat
```
安装脚本将：
- 以 TRELLIS 环境为基础进行设置。
- 安装所有模型依赖 (PartField, Hunyuan3D, HoloPart, UniRig, PartPacker)。
- 安装 FastAPI 和后端依赖。

3. **下载预训练模型(可选，或自动下载)**:
```bash
# Linux 系统
chmod +x download_models.sh
# 下载特定模型
./scripts/download_models.sh -m partfield,trellis
# 验证所有现有模型
./download_models.sh -v
# 显示帮助
./download_models.sh -h
# 列出可用模型
./download_models.sh --list

# Windows 系统
# 下载特定模型
.\scripts\download_models.bat -m partfield,trellis
# 验证所有现有模型
.\scripts\download_models.bat -v
# 显示帮助
.\scripts\download_models.bat -h
# 列出可用模型
.\scripts\download_models.bat --list
```

### 运行服务器
```bash
# Linux 系统
# 开发模式 (自动重载)
chmod +x scripts/run_server.sh
P3D_RELOAD=true ./scripts/run_server.sh
# 生产模式，(注意您还需要更改 .yaml 规范)
P3D_RELOAD=false ./scripts/run_server.sh
# 自定义配置
P3D_HOST=0.0.0.0 P3D_PORT=7842 P3D_WORKERS=4 ./scripts/run_server.sh

# Windows 系统  
# 开发模式 (自动重载)
set P3D_RELOAD=true
.\scripts\run_server.bat
# 生产模式
set P3D_RELOAD=false
.\scripts\run_server.bat
# 自定义配置
set P3D_HOST=0.0.0.0
set P3D_PORT=7842
set P3D_WORKERS=4
.\scripts\run_server.bat
```
默认情况下，服务器将在 `http://localhost:7842` 上可用。

### API 文档

服务器运行后，访问：
- **交互式 API 文档**: `http://localhost:7842/docs`
- **ReDoc 文档**: `http://localhost:7842/redoc`

## 📖 使用示例
### 基础操作
```bash 
# 检查系统状态
curl -X GET "http://localhost:7842/api/v1/system/status/"
# 检查可用功能
curl -X GET "http://localhost:7842/api/v1/system/features"
# 检查可用模型
curl -X GET "http://localhost:7842/api/v1/system/models"
```
<details>
<summary>查询功能的示例响应</summary>

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

### 文本到3D网格
```bash
curl -X POST "http://localhost:7842/api/v1/mesh-generation/text-to-textured-mesh" \
  -H "Content-Type: application/json" \
  -d '{
    "text_prompt": "A cute robot cat",
    "output_format": "glb",
    "model_preference": "trellis_text_to_textured_mesh"
  }'
```

### 图像到3D网格
```bash
curl -X POST "http://localhost:7842/api/v1/mesh-generation/image-to-textured-mesh" \
  -F "image_file=@assets/example_image/typical_humanoid_mech.png" \
  -F "output_format=glb"
```

### 网格分割
```bash
curl -X POST "http://localhost:7842/api/v1/mesh-segmentation/segment-mesh" \
  -F "mesh_file=@assets/example_mesh/typical_creature_dragon.obj" \
  -F "model_preference=partfield_mesh_segmentation"
```

### 自动绑定
```bash
curl -X POST "http://localhost:7842/api/v1/auto-rigging/generate-rig" \
  -F "rig_mode=full" \
  -F "mesh_file=@assets/example_autorig/giraffe.glb" \
  -F "output_format=fbx"
```

## 🧪 测试

### 适配器测试
```bash
# 测试所有适配器
python tests/run_adapter_tests.py 
# 测试特定适配器，例如
PYTHONPATH=. pytest tests/test_adapters/test_trellis_adapter.py -v -s -r s
```

### 集成测试
```bash
# 提交作业，等待完成，然后下一个作业
python tests/run_test_client.py --server-url http://localhost:7842 \
  --timeout 600 --poll-interval 25 --output-dir test_results.sequential --sequential 

# 一次提交所有作业，然后监控所有作业 (默认行为)
# 超时时间更长以覆盖所有作业
python tests/run_test_client.py --server-url http://localhost:7842 \
  --timeout 3600 --poll-interval 30 --output-dir test_results.concurrent 
```

### 其他测试
```bash
# 测试基本 API 端点
pytest tests/test_basic_endpoints.py -v -s -r s
# 运行按需多进程调度器
python tests/test_on_demand_scheduler.py 
```

## ⚙️ 配置
* [系统配置](./config/system.yaml)
* [模型配置](./config/models.yaml)
* [日志配置](./config/logging.yaml)

## 🔧 开发
### 项目结构
```
3DAIGC-API/
├── api/                   # FastAPI 应用
│   ├── main.py            # 入口点
│   └── routers/           # API 端点
├── adapters/              # 模型适配器
├── core/                  # 核心框架
│   ├── scheduler/         # GPU 调度
│   └── models/            # 模型抽象
├── config/                # 配置文件
├── tests/                 # 测试套件
├── thirdparty/            # 第三方模型
└── utils/                 # 工具
```

### 添加新模型
1. 在 `adapters/` 中按照基础接口创建适配器
2. 在 `config/models.yaml` 和模型工厂 `core/scheduler/model_factory.py` 中注册模型
3. 在 `tests` 中添加适配器测试和/或集成测试

## 注意事项
1. 当前系统**仅支持单个 UVICORN WORKER**，因为多个 worker 会有多个调度器实例并破坏所有功能。但由于 AI 推理通常是 GPU 密集型的，所以在小规模使用中**不会成为瓶颈**，因为 worker 仅用于处理传入的 web 请求。
2. 频繁加载/卸载模型可能会很慢（您可能会在测试客户端中观察到）。在实践中最好只启用所需的模型并始终将它们保留在显存中。
3. 大量代码是用 vibe coding (Cursor + Claude4) 编写的，Claude4 是一个好的软件工程师，作为系统设计的初学者，我从他/她那里学到了很多。如果您也感兴趣，可以查看 [vibe coding prompt](./docs/vibe_coding_prompt.md) 和 [vibe coding READMEs](./docs/vibe_coding/)。

## 🛣️ 待办事项
### 短期
- [ ] 更好地组织（清理）当前 API 服务的输出目录
- [ ] 支持多视图图像作为网格生成模型的条件
- [ ] 暴露并支持更多参数（例如网格生成中的抽取比率）

### 长期
- [ ] Windows 一键安装程序
- [ ] 作业队列和调度器切换到 redis，可能添加速率限制
- [ ] 基于这个 3D API 集合，复制/实现类似 Tripo/Hunyuan 的 3D 工作室，其中前端和后端都可以轻松部署在个人电脑上

## 📄 许可证
此项目基于 Apache License 2.0 许可 - 详情请参阅 [LICENSE](LICENSE) 文件。
注意，每个算法**都有其自己的许可证**，如果需要请仔细检查。

## 🌟 致谢
特别感谢所有集成模型的作者和贡献者以及开源社区！