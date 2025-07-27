#!/bin/bash

set -e  # Exit on any error

echo "========================================"
echo "3DAIGC-API Docker Build Script"
echo "========================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed"
}

# Check NVIDIA runtime
check_nvidia_runtime() {
    print_status "Checking NVIDIA runtime..."
    if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_error "NVIDIA runtime not working. Please install nvidia-container-toolkit."
        exit 1
    fi
    print_success "NVIDIA runtime is working"
}

# Initialize git submodules
init_submodules() {
    print_status "Initializing git submodules..."
    if [ -f ".gitmodules" ]; then
        git submodule update --init --recursive
        print_success "Git submodules initialized"
    else
        print_warning "No .gitmodules file found, skipping submodule initialization"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data uploads models logs
    print_success "Directories created"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    docker-compose build
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Optional: Download models
download_models() {
    read -p "Do you want to download pre-trained models? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Downloading models..."
        if [ -f "scripts/download_models.sh" ]; then
            bash scripts/download_models.sh
            print_success "Models downloaded"
        else
            print_warning "Download script not found, skipping model download"
        fi
    fi
}

# Main execution
main() {
    echo "Starting Docker build process..."
    echo ""
    
    # Perform checks
    check_docker
    check_docker_compose
    check_nvidia_runtime
    
    # Setup
    init_submodules
    create_directories
    
    # Build
    build_image
    
    # Optional model download
    download_models
    
    echo ""
    echo "========================================"
    print_success "Build completed successfully!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. Start the service: docker-compose up -d"
    echo "2. Check status: docker-compose ps"
    echo "3. View logs: docker-compose logs -f 3daigc-api"
    echo "4. Access API: http://localhost:8000"
    echo "5. API docs: http://localhost:8000/docs"
    echo ""
    echo "For more information, see DOCKER_README.md"
}

# Execute main function
main "$@" 