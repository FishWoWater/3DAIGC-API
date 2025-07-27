# 3DAIGC-API Docker Deployment Guide

This guide provides instructions for building and deploying the 3DAIGC-API using Docker.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support
- **Memory**: Minimum 16GB RAM (32GB+ recommended)
- **Storage**: At least 50GB free space
- **Docker**: Docker Engine 20.10+
- **Docker Compose**: Version 2.0+

### Software Dependencies
1. **Docker with NVIDIA Runtime Support**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Verify GPU Access**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
   ```

## Building the Docker Image

### Method 1: Using Docker Compose (Recommended)

1. **Clone the repository and navigate to the project directory**
   ```bash
   git clone <your-repo-url>
   cd 3DAIGC-API
   ```

2. **Initialize git submodules**
   ```bash
   git submodule update --init --recursive
   ```

3. **Create necessary directories**
   ```bash
   mkdir -p data uploads models logs
   ```

4. **Build and run with Docker Compose**
   ```bash
   # Build the image
   docker-compose build

   # Start the service
   docker-compose up -d

   # View logs
   docker-compose logs -f 3daigc-api
   ```

### Method 2: Using Docker CLI

1. **Build the Docker image**
   ```bash
   docker build -t 3daigc-api:latest .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     --name 3daigc-api \
     --gpus all \
     -p 8000:8000 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/uploads:/app/uploads \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/logs:/app/logs \
     -e PYTHONPATH=/app \
     -e CUDA_VISIBLE_DEVICES=0 \
     --restart unless-stopped \
     3daigc-api:latest
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONPATH` | Python module search path | `/app` |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | `0` |
| `TORCH_CUDA_ARCH_LIST` | CUDA architectures to support | `"6.0 6.1 7.0 7.5 8.0 8.6"` |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Data storage |
| `./uploads` | `/app/uploads` | File uploads |
| `./models` | `/app/models` | Model files |
| `./logs` | `/app/logs` | Application logs |

## Usage

### Starting the Service

```bash
# Using Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f 3daigc-api
```

### Accessing the API

Once the container is running, you can access:

- **API Endpoints**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# Upload a file for processing
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_file.png"

# Generate 3D mesh from image
curl -X POST "http://localhost:8000/generate/mesh" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "path/to/image.png", "prompt": "a 3D model"}'
```

## Monitoring and Debugging

### Container Management

```bash
# Check container status
docker-compose ps

# View real-time logs
docker-compose logs -f 3daigc-api

# Execute commands in container
docker-compose exec 3daigc-api bash

# Restart service
docker-compose restart 3daigc-api

# Stop all services
docker-compose down
```

### Resource Monitoring

```bash
# Monitor container resource usage
docker stats 3daigc-api

# Check GPU usage inside container
docker-compose exec 3daigc-api nvidia-smi

# Monitor disk usage
docker system df
```

### Troubleshooting

#### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Verify NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
   
   # Check container GPU access
   docker-compose exec 3daigc-api nvidia-smi
   ```

2. **Memory Issues**
   ```bash
   # Check available memory
   docker-compose exec 3daigc-api free -h
   
   # Monitor GPU memory
   docker-compose exec 3daigc-api nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   ```

3. **Permission Issues**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER data uploads models logs
   ```

4. **Build Failures**
   ```bash
   # Clean build cache
   docker system prune -a
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

## Production Deployment

### Using Reverse Proxy (Nginx)

1. **Create nginx.conf**
   ```nginx
   events {
       worker_connections 1024;
   }
   
   http {
       upstream 3daigc-api {
           server 3daigc-api:8000;
       }
   
       server {
           listen 80;
           server_name your-domain.com;
   
           client_max_body_size 100M;
   
           location / {
               proxy_pass http://3daigc-api;
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
               proxy_set_header X-Forwarded-Proto $scheme;
               proxy_read_timeout 300s;
               proxy_connect_timeout 75s;
           }
       }
   }
   ```

2. **Start with production profile**
   ```bash
   docker-compose --profile production up -d
   ```

### Performance Optimization

1. **Multi-GPU Support**
   ```bash
   # Modify docker-compose.yml
   environment:
     - CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

2. **Memory Optimization**
   ```bash
   # Add memory limits
   deploy:
     resources:
       limits:
         memory: 32G
   ```

## Backup and Maintenance

### Data Backup

```bash
# Backup data volumes
docker run --rm -v 3daigc-api_data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# Restore data
docker run --rm -v 3daigc-api_data:/data -v $(pwd):/backup alpine tar xzf /backup/data-backup.tar.gz -C /data
```

### Updates

```bash
# Pull latest changes
git pull origin main
git submodule update --recursive

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Security Considerations

1. **Network Security**
   - Use reverse proxy with SSL/TLS
   - Restrict access with firewall rules
   - Use environment files for secrets

2. **Container Security**
   - Run containers as non-root user
   - Use read-only volumes where possible
   - Regularly update base images

3. **API Security**
   - Implement authentication/authorization
   - Rate limiting
   - Input validation

## Support

For issues and questions:
- Check the logs: `docker-compose logs -f 3daigc-api`
- Verify system requirements
- Consult the main project README.md
- Open an issue in the project repository 