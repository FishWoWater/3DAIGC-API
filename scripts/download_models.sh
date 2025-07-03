#!/bin/bash

# 3DAIGC Model Download Script
# Usage: ./download_models.sh [OPTIONS]
# 
# Available models:
#   partfield, hunyuan2, hunyuan2mini, hunyuan21, trellis, trellis-text, 
#   holopart, unirig, partpacker, misc, all
#
# Options:
#   -h, --help              Show this help message
#   -m, --models MODEL      Comma-separated list of models to download (default: all)
#   -v, --verify            Verify existing models without downloading
#   -f, --force             Force re-download even if files exist
#   --list                  List all available models

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
MODELS_TO_DOWNLOAD="all"
VERIFY_ONLY=false
FORCE_DOWNLOAD=false

# Available models
AVAILABLE_MODELS=("partfield" "hunyuan2" "hunyuan2mini" "hunyuan21" "trellis" "trellis-text" "holopart" "unirig" "partpacker" "misc" "all")

show_help() {
    cat << EOF
3DAIGC Model Download Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -m, --models MODELS     Comma-separated list of models to download (default: all)
    -v, --verify            Verify existing models without downloading
    -f, --force             Force re-download even if files exist
    --list                  List all available models

Available models:
    partfield     - PartField model for mesh segmentation
    hunyuan2      - Hunyuan3D 2.0 models (geometry/texture/vae)
    hunyuan2mini  - Hunyuan3D 2.0 mini models
    hunyuan21     - Hunyuan3D 2.1 models  
    trellis       - TRELLIS image-large model
    trellis-text  - TRELLIS text-xlarge model (optional)
    holopart      - HoloPart model for part completion
    unirig        - UniRig model for auto-rigging
    partpacker    - PartPacker model
    misc          - Miscellaneous models (RealESRGAN, DINOv2)
    all           - Download all models

Examples:
    $0                                    # Download all models
    $0 -m hunyuan2,trellis               # Download only Hunyuan3D 2.0 and TRELLIS
    $0 -v                                # Verify all existing models
    $0 -m partfield -f                   # Force re-download PartField model
    $0 --list                           # List available models

EOF
}

list_models() {
    echo "Available models:"
    for model in "${AVAILABLE_MODELS[@]}"; do
        if [ "$model" != "all" ]; then
            echo "  - $model"
        fi
    done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--models)
            MODELS_TO_DOWNLOAD="$2"
            shift 2
            ;;
        -v|--verify)
            VERIFY_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --list)
            list_models
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Function to check if file exists and get its size
verify_file() {
    local file_path="$1"
    local min_size="${2:-1000}"  # Minimum size in bytes (default 1KB)
    
    if [ -f "$file_path" ]; then
        local file_size=$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null || echo "0")
        if [ "$file_size" -gt "$min_size" ]; then
            print_success "✓ $file_path ($(numfmt --to=iec-i --suffix=B $file_size))"
            return 0
        else
            print_warning "✗ $file_path exists but is too small ($(numfmt --to=iec-i --suffix=B $file_size))"
            return 1
        fi
    else
        print_warning "✗ $file_path not found"
        return 1
    fi
}

# Function to verify directory exists and has content
verify_directory() {
    local dir_path="$1"
    local min_files="${2:-1}"
    
    if [ -d "$dir_path" ]; then
        local file_count=$(find "$dir_path" -type f | wc -l)
        if [ "$file_count" -ge "$min_files" ]; then
            print_success "✓ $dir_path ($file_count files)"
            return 0
        else
            print_warning "✗ $dir_path exists but has insufficient files ($file_count files, need $min_files)"
            return 1
        fi
    else
        print_warning "✗ $dir_path not found"
        return 1
    fi
}

# Function to download with verification
download_with_verify() {
    local url="$1"
    local output_path="$2"
    local description="$3"
    
    print_info "Downloading $description..."
    print_info "URL: $url"
    print_info "Output: $output_path"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$output_path")"
    
    # Download the file
    if wget -O "$output_path" "$url"; then
        # Verify the download
        if verify_file "$output_path"; then
            print_success "Successfully downloaded $description"
        else
            print_error "Downloaded file verification failed for $description"
            return 1
        fi
    else
        print_error "Failed to download $description"
        return 1
    fi
}

# Function to download PartField model
download_partfield() {
    print_info "========================================"
    print_info "Downloading PartField Model"
    print_info "========================================"
    
    local model_path="pretrained/PartField/model_objaverse.pt"
    
    if [ "$FORCE_DOWNLOAD" = false ] && verify_file "$model_path" 50000000; then # 50MB minimum
        print_info "PartField model already exists and verified"
        return 0
    fi
    
    mkdir -p pretrained/PartField
    download_with_verify \
        "https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt" \
        "$model_path" \
        "PartField model"
}

# Function to download Hunyuan3D 2.0 models
download_hunyuan2() {
    print_info "========================================"
    print_info "Downloading Hunyuan3D 2.0 Models"
    print_info "========================================"
    
    local model_dir="pretrained/tencent/Hunyuan3D-2"
    
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$model_dir" 10; then
        print_info "Hunyuan3D 2.0 models already exist and verified"
        return 0
    fi
    
    mkdir -p "$model_dir"
    print_info "Downloading Hunyuan3D 2.0 (geometry/texture/vae)..."
    print_info "Notice that the paint and delight models are ONLY needed when you need to texture generation feature"
    if huggingface-cli download  tencent/Hunyuan3D-2 \
        --include "hunyuan3d-dit-v2-0-turbo/*" "hunyuan3d-vae-v2-0-turbo/*" "hunyuan3d-paint-v2-0-turbo/*" "hunyuan3d-delight-v2-0/*" \
        --local-dir "$model_dir"; then
        print_success "Hunyuan3D 2.0 models downloaded successfully"
    else
        print_error "Failed to download Hunyuan3D 2.0 models"
        return 1
    fi
}

# Function to download Hunyuan3D 2.0 mini models
download_hunyuan2mini() {
    print_info "========================================"
    print_info "Downloading Hunyuan3D 2.0 Mini Models"
    print_info "========================================"
    
    local model_dir="pretrained/tencent/Hunyuan3D-2mini"
    
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$model_dir" 5; then
        print_info "Hunyuan3D 2.0 mini models already exist and verified"
        return 0
    fi
    
    mkdir -p "$model_dir"
    print_info "Downloading Hunyuan3D 2.0 mini (geometry/vae)..."
    if huggingface-cli download  tencent/Hunyuan3D-2mini \
        --include "hunyuan3d-dit-v2-mini-turbo/*" "hunyuan3d-vae-v2-mini-turbo/*" \
        --local-dir "$model_dir"; then
        print_success "Hunyuan3D 2.0 mini models downloaded successfully"
    else
        print_error "Failed to download Hunyuan3D 2.0 mini models"
        return 1
    fi
}

# Function to download Hunyuan3D 2.1 models
download_hunyuan21() {
    print_info "========================================"
    print_info "Downloading Hunyuan3D 2.1 Models"
    print_info "========================================"
    
    local model_dir="pretrained/tencent/Hunyuan3D-2.1"
    
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$model_dir" 5; then
        print_info "Hunyuan3D 2.1 models already exist and verified"
        return 0
    fi
    
    mkdir -p "$model_dir"
    print_info "Downloading Hunyuan3D 2.1 models..."
    if huggingface-cli download  tencent/Hunyuan3D-2.1 --local-dir "$model_dir"; then
        print_success "Hunyuan3D 2.1 models downloaded successfully"
    else
        print_error "Failed to download Hunyuan3D 2.1 models"
        return 1
    fi
}

# Function to download TRELLIS models
download_trellis() {
    print_info "========================================"
    print_info "Downloading TRELLIS Image-Large Model"
    print_info "========================================"
    
    local model_dir="pretrained/TRELLIS/TRELLIS-image-large"
    
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$model_dir" 5; then
        print_info "TRELLIS image-large model already exists and verified"
        return 0
    fi
    
    mkdir -p "$model_dir"
    print_info "Downloading TRELLIS image-large model..."
    if huggingface-cli download  microsoft/TRELLIS-image-large --local-dir "$model_dir"; then
        print_success "TRELLIS image-large model downloaded successfully"
    else
        print_error "Failed to download TRELLIS image-large model"
        return 1
    fi
}

# Function to download TRELLIS text model (optional)
download_trellis_text() {
    print_info "========================================"
    print_info "Downloading TRELLIS Text-XLarge Model"
    print_info "========================================"
    
    local model_dir="pretrained/TRELLIS/TRELLIS-text-xlarge"
    
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$model_dir" 5; then
        print_info "TRELLIS text-xlarge model already exists and verified"
        return 0
    fi
    
    mkdir -p "$model_dir"
    print_info "Downloading TRELLIS text-xlarge model (optional, for text-conditioned part re-texturing)..."
    if huggingface-cli download  microsoft/TRELLIS-text-xlarge --local-dir "$model_dir"; then
        print_success "TRELLIS text-xlarge model downloaded successfully"
    else
        print_error "Failed to download TRELLIS text-xlarge model"
        return 1
    fi
}

# Function to download HoloPart model
download_holopart() {
    print_info "========================================"
    print_info "Downloading HoloPart Model"
    print_info "========================================"
    
    local model_dir="pretrained/HoloPart"
    
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$model_dir" 3; then
        print_info "HoloPart model already exists and verified"
        return 0
    fi
    
    mkdir -p "$model_dir"
    print_info "Downloading HoloPart model..."
    if huggingface-cli download  VAST-AI/HoloPart --local-dir "$model_dir"; then
        print_success "HoloPart model downloaded successfully"
    else
        print_error "Failed to download HoloPart model"
        return 1
    fi
}

# Function to download UniRig model
download_unirig() {
    print_info "========================================"
    print_info "Downloading UniRig Model"
    print_info "========================================"
    
    local model_dir="pretrained/UniRig"
    
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$model_dir" 3; then
        print_info "UniRig model already exists and verified"
        return 0
    fi
    
    mkdir -p "$model_dir"
    print_info "Downloading UniRig model..."
    if huggingface-cli download  VAST-AI/UniRig --local-dir "$model_dir"; then
        print_success "UniRig model downloaded successfully"
    else
        print_error "Failed to download UniRig model"
        return 1
    fi
}

# Function to download PartPacker model
download_partpacker() {
    print_info "========================================"
    print_info "Downloading PartPacker Model"
    print_info "========================================"
    
    local model_dir="pretrained/PartPacker"
    
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$model_dir" 3; then
        print_info "PartPacker model already exists and verified"
        return 0
    fi
    
    mkdir -p "$model_dir"
    print_info "Downloading PartPacker model..."
    if huggingface-cli download  nvidia/PartPacker --local-dir "$model_dir"; then
        print_success "PartPacker model downloaded successfully"
    else
        print_error "Failed to download PartPacker model"
        return 1
    fi
}

# Function to download miscellaneous models
download_misc() {
    print_info "========================================"
    print_info "Downloading Miscellaneous Models"
    print_info "========================================"
    
    # RealESRGAN_x4plus for Hunyuan3D-2.1
    local realesrgan_path="pretrained/misc/RealESRGAN_x4plus.pth"
    if [ "$FORCE_DOWNLOAD" = false ] && verify_file "$realesrgan_path" 50000000; then # 50MB minimum
        print_info "RealESRGAN_x4plus already exists and verified"
    else
        mkdir -p pretrained/misc
        download_with_verify \
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
            "$realesrgan_path" \
            "RealESRGAN_x4plus model"
    fi
    
    # DINOv2-giant used in PartPacker or elsewhere
    local dinov2_dir="pretrained/dinov2-giant"
    if [ "$FORCE_DOWNLOAD" = false ] && verify_directory "$dinov2_dir" 5; then
        print_info "DINOv2-giant model already exists and verified"
    else
        mkdir -p "$dinov2_dir"
        print_info "Downloading DINOv2-giant model..."
        if huggingface-cli download  facebook/dinov2-giant \
            --local-dir "$dinov2_dir" --exclude "*.bin"; then
            print_success "DINOv2-giant model downloaded successfully"
        else
            print_error "Failed to download DINOv2-giant model"
            return 1
        fi
    fi
}

# Function to verify all models
verify_all_models() {
    print_info "========================================"
    print_info "Verifying All Models"
    print_info "========================================"
    
    local all_verified=true
    
    print_info "Checking PartField..."
    verify_file "pretrained/PartField/model_objaverse.pt" 50000000 || all_verified=false
    
    print_info "Checking Hunyuan3D 2.0..."
    verify_directory "pretrained/tencent/Hunyuan3D-2" 10 || all_verified=false
    
    print_info "Checking Hunyuan3D 2.0 mini..."
    verify_directory "pretrained/tencent/Hunyuan3D-2mini" 5 || all_verified=false
    
    print_info "Checking Hunyuan3D 2.1..."
    verify_directory "pretrained/tencent/Hunyuan3D-2.1" 5 || all_verified=false
    
    print_info "Checking TRELLIS image-large..."
    verify_directory "pretrained/TRELLIS/TRELLIS-image-large" 5 || all_verified=false
    
    print_info "Checking TRELLIS text-xlarge (optional)..."
    verify_directory "pretrained/TRELLIS/TRELLIS-text-xlarge" 5 || print_warning "TRELLIS text-xlarge not found (optional)"
    
    print_info "Checking HoloPart..."
    verify_directory "pretrained/HoloPart" 3 || all_verified=false
    
    print_info "Checking UniRig..."
    verify_directory "pretrained/UniRig" 3 || all_verified=false
    
    print_info "Checking PartPacker..."
    verify_directory "pretrained/PartPacker" 3 || all_verified=false
    
    print_info "Checking miscellaneous models..."
    verify_file "pretrained/misc/RealESRGAN_x4plus.pth" 50000000 || all_verified=false
    verify_directory "pretrained/dinov2-giant" 5 || all_verified=false
    
    if [ "$all_verified" = true ]; then
        print_success "All required models are present and verified!"
    else
        print_warning "Some models are missing or corrupted. Run without -v flag to download them."
    fi
}

# Main execution
print_info "========================================"
print_info "3DAIGC Model Download Script"
print_info "========================================"

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    print_error "huggingface-cli is not installed. Please install it first:"
    print_error "pip install huggingface_hub"
    exit 1
fi

# Check if wget is available
if ! command -v wget &> /dev/null; then
    print_error "wget is not installed. Please install it first."
    exit 1
fi

# If verify only mode
if [ "$VERIFY_ONLY" = true ]; then
    verify_all_models
    exit 0
fi

# Parse models to download
IFS=',' read -ra MODELS_ARRAY <<< "$MODELS_TO_DOWNLOAD"

# Download requested models
for model in "${MODELS_ARRAY[@]}"; do
    case "$model" in
        "partfield")
            download_partfield
            ;;
        "hunyuan2")
            download_hunyuan2
            ;;
        "hunyuan2mini")
            download_hunyuan2mini
            ;;
        "hunyuan21")
            download_hunyuan21
            ;;
        "trellis")
            download_trellis
            ;;
        "trellis-text")
            download_trellis_text
            ;;
        "holopart")
            download_holopart
            ;;
        "unirig")
            download_unirig
            ;;
        "partpacker")
            download_partpacker
            ;;
        "misc")
            download_misc
            ;;
        "all")
            download_partfield
            download_hunyuan2
            download_hunyuan2mini
            download_hunyuan21
            download_trellis
            download_trellis_text
            download_holopart
            download_unirig
            download_partpacker
            download_misc
            ;;
        *)
            print_error "Unknown model: $model"
            print_error "Available models: $(IFS=', '; echo "${AVAILABLE_MODELS[*]}")"
            exit 1
            ;;
    esac
done

print_success "========================================"
print_success "Model Download Complete!"
print_success "========================================"
print_info "All requested models have been downloaded successfully."
print_info "You can verify the downloads by running: $0 -v"