#!/bin/bash

echo "========================================"
echo "Starting 3DAIGC-API Installation"
echo "========================================"
echo "The installation may take a while, please wait..."
echo ""

echo "[INFO] Creating conda environment '3daigc-api' with Python 3.10..."
conda create -n 3daigc-api python=3.10 -y
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Conda environment created successfully"
else
    echo "[ERROR] Failed to create conda environment"
    exit 1
fi

echo "[INFO] Activating conda environment..."
conda activate 3daigc-api

echo "[INFO] Installing PyTorch with CUDA 12.1 support..."
## install pytorch for specific cuda versions
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PyTorch installation completed"
else
    echo "[ERROR] Failed to install PyTorch"
    exit 1
fi

echo ""
echo "========================================"
echo "Installing TRELLIS Dependencies"
echo "========================================"
### we startup with the environment of trellis ###
echo "[INFO] Changing directory to thirdparty/TRELLIS..."
cd thirdparty/TRELLIS
echo "[INFO] Running TRELLIS setup script..."
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TRELLIS setup completed"
else
    echo "[ERROR] TRELLIS setup failed"
    exit 1
fi
# for systems with glibc < 2.29 , you may need to build kaolin from source manually
echo "[NOTE] For systems with glibc < 2.29, you may need to build kaolin from source manually"

echo ""
echo "========================================"
echo "Installing PartField Dependencies"
echo "========================================"
# install PartField for mesh segmentation 
echo "[INFO] Changing directory to thirdparty/PartField..."
cd ../../thirdparty/PartField 
echo "[INFO] Installing PartField core dependencies..."
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PartField core dependencies installed"
else
    echo "[ERROR] Failed to install PartField core dependencies"
    exit 1
fi

echo "[INFO] Installing additional PartField dependencies..."
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d psutil 
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Additional PartField dependencies installed"
else
    echo "[ERROR] Failed to install additional PartField dependencies"
    exit 1
fi

echo "[INFO] Installing PyTorch Geometric extensions..."
pip install torch-scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PyTorch Geometric extensions installed"
else
    echo "[ERROR] Failed to install PyTorch Geometric extensions"
    exit 1
fi
# installation for PartField end 
echo "[SUCCESS] PartField installation completed"

echo ""
echo "========================================"
echo "Installing Hunyuan3D 2.0 Dependencies"
echo "========================================"
### install hunyuan3d for mesh generation  ###
echo "[INFO] Changing directory to thirdparty/Hunyuan3D-2..."
cd ../../thirdparty/Hunyuan3D-2
echo "[INFO] Installing Hunyuan3D 2.0 requirements..."
pip install -r requirements.txt 
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Hunyuan3D 2.0 requirements installed"
else
    echo "[ERROR] Failed to install Hunyuan3D 2.0 requirements"
    exit 1
fi

echo "[INFO] Installing Hunyuan3D 2.0 package..."
pip install -e .
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Hunyuan3D 2.0 package installed"
else
    echo "[ERROR] Failed to install Hunyuan3D 2.0 package"
    exit 1
fi

echo "[INFO] Building custom rasterizer..."
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Custom rasterizer built successfully"
else
    echo "[ERROR] Failed to build custom rasterizer"
    exit 1
fi

echo "[INFO] Building differentiable renderer..."
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
cd ../../../
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Differentiable renderer built successfully"
else
    echo "[ERROR] Failed to build differentiable renderer"
    exit 1
fi
### installation for hunyuan3d 2.0 end ###
echo "[SUCCESS] Hunyuan3D 2.0 installation completed"

echo ""
echo "========================================"
echo "Installing Hunyuan3D 2.1 Dependencies"
echo "========================================"
### installation for hunyuan3d 2.1  ###
echo "[INFO] Changing directory to thirdparty/Hunyuan3D-2.1..."
cd ../../thirdparty/Hunyuan3D-2.1
echo "[INFO] Installing custom rasterizer for Hunyuan3D 2.1..."
cd hy3dpaint/custom_rasterizer
pip install -e .
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Hunyuan3D 2.1 custom rasterizer installed"
else
    echo "[ERROR] Failed to install Hunyuan3D 2.1 custom rasterizer"
    exit 1
fi

echo "[INFO] Building differentiable renderer for Hunyuan3D 2.1..."
cd ../..
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Hunyuan3D 2.1 differentiable renderer built successfully"
else
    echo "[ERROR] Failed to build Hunyuan3D 2.1 differentiable renderer"
    exit 1
fi
cd ../..
echo "[INFO] Installing Hunyuan3D 2.1 requirements..."
pip install -r requirements-inference.txt 
### installation for hunyuan3d 2.1 end ###
echo "[SUCCESS] Hunyuan3D 2.1 installation completed"

echo ""
echo "========================================"
echo "Installing HoloPart Dependencies"
echo "========================================"
### holopart for part completion  ###
echo "[INFO] Changing directory to thirdparty/HoloPart..."
cd ../../thirdparty/HoloPart
echo "[INFO] Installing HoloPart requirements..."
pip install -r requirements.txt 
if [ $? -eq 0 ]; then
    echo "[SUCCESS] HoloPart requirements installed"
else
    echo "[ERROR] Failed to install HoloPart requirements"
    exit 1
fi
### holopart for part completion end ###
echo "[SUCCESS] HoloPart installation completed"

echo ""
echo "========================================"
echo "Installing UniRig Dependencies"
echo "========================================"
### unirig for auto-rigging  ###
echo "[INFO] Changing directory to thirdparty/UniRig..."
cd ../../thirdparty/UniRig
echo "[INFO] Installing spconv-cu120 for UniRig..."
pip install spconv-cu120
pip install pyrender fast-simplification python-box timm
if [ $? -eq 0 ]; then
    echo "[SUCCESS] UniRig dependencies installed"
else
    echo "[ERROR] Failed to install UniRig dependencies"
    exit 1
fi

echo ""
echo "========================================"
echo "Installing PartPacker Dependencies"
echo "========================================"
### part packer  ###
echo "[INFO] Changing directory to thirdparty/PartPacker..."
cd ../../thirdparty/PartPacker
echo "[INFO] Installing PartPacker requirements..."
pip install meshiki fpsample kiui pymcubes einops
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PartPacker requirements installed"
else
    echo "[ERROR] Failed to install PartPacker requirements"
    exit 1
fi
### part packer end ###
echo "[SUCCESS] PartPacker installation completed"

cd ../../

echo ""
echo "========================================"
echo "Installing Project Dependencies"
echo "========================================"
### for this project (fastapi / uvicorn relevant etc.)  ###
echo "[INFO] Installing main project requirements..."
pip install -r requirements.txt 
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Main project requirements installed"
else
    echo "[ERROR] Failed to install main project requirements"
    exit 1
fi

echo "[INFO] Installing test requirements..."
# testing 
pip install -r requirements-test.txt 
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Test requirements installed"
else
    echo "[ERROR] Failed to install test requirements"
    exit 1
fi

echo "[INFO] Installing huggingface_hub for model downloading..."
# for downloading models 
pip install huggingface_hub
if [ $? -eq 0 ]; then
    echo "[SUCCESS] huggingface_hub installed"
else
    echo "[ERROR] Failed to install huggingface_hub"
    exit 1
fi

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo "All installation done successfully!"

echo "Checking CUDA availability..."
python -c "import torch; print(torch.cuda.is_available())" && echo "CUDA installed successfully" || echo "Failed"

echo "Checking PyTorch version..."
python -c "import torch; print(torch.__version__)" && echo "PyTorch installed successfully" || echo "Failed"

echo "Checking Blender availability..."
python -c "import bpy" && echo "Blender installed successfully" || echo "Failed"

echo "Checking Other Packages..."
python -c "import kaolin; print(kaolin.__version__)" && echo "Kaolin installed successfully" || echo "Failed"
python -c "import open3d; import pymeshlab" && echo "Open3D and pymeshlab installed successfully" || echo "Failed"

# install other runtime dependencies
sudo apt update
sudo apt install libsm6 libegl1 libegl1-mesa libgl1-mesa-dev -y # for rendering 



