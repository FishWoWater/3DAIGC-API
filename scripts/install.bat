@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Starting 3DAIGC-API Installation
echo ========================================
echo The installation may take a while, please wait...
echo.

echo [INFO] Creating conda environment '3daigc-api' with Python 3.10...
call conda create -n 3daigc-api python=3.10 -y
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Conda environment created successfully
) else (
    echo [ERROR] Failed to create conda environment
    exit /b 1
)

echo [INFO] Activating conda environment...
call conda activate 3daigc-api


echo.
echo ========================================
echo Installing TRELLIS Dependencies
echo ========================================
REM we startup with the environment of trellis
echo [INFO] Changing directory to thirdparty\TRELLIS...
cd thirdparty\TRELLIS
echo [INFO] Running TRELLIS setup script...
REM Note: Windows doesn't have direct equivalent of bash sourcing, so we'll run pip commands directly
@REM The instruction is from https://github.com/microsoft/TRELLIS/issues/3
call .\scripts\install_trellis.bat

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] TRELLIS setup completed
) else (
    echo [ERROR] TRELLIS setup failed
    exit /b 1
)
REM for systems with glibc < 2.29 , you may need to build kaolin from source manually
echo [NOTE] For systems with older libraries, you may need to build kaolin from source manually

echo.
echo ========================================
echo Installing PartField Dependencies
echo ========================================
REM install PartField for mesh segmentation 
echo [INFO] Changing directory to thirdparty\PartField...
cd ..\..\thirdparty\PartField 
echo [INFO] Installing PartField core dependencies...
call pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] PartField core dependencies installed
) else (
    echo [ERROR] Failed to install PartField core dependencies
    exit /b 1
)

echo [INFO] Installing additional PartField dependencies...
call pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d psutil 
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Additional PartField dependencies installed
) else (
    echo [ERROR] Failed to install additional PartField dependencies
    exit /b 1
)

echo [INFO] Installing PyTorch Geometric extensions...
call pip install torch-scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] PyTorch Geometric extensions installed
) else (
    echo [ERROR] Failed to install PyTorch Geometric extensions
    exit /b 1
)
REM installation for PartField end 
echo [SUCCESS] PartField installation completed

echo.
echo ========================================
echo Installing Hunyuan3D 2.0 Dependencies
echo ========================================
REM install hunyuan3d for mesh generation
echo [INFO] Changing directory to thirdparty\Hunyuan3D-2...
cd ..\..\thirdparty\Hunyuan3D-2
echo [INFO] Installing Hunyuan3D 2.0 requirements...
call pip install -r requirements.txt 
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Hunyuan3D 2.0 requirements installed
) else (
    echo [ERROR] Failed to install Hunyuan3D 2.0 requirements
    exit /b 1
)

echo [INFO] Installing Hunyuan3D 2.0 package...
call pip install -e .
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Hunyuan3D 2.0 package installed
) else (
    echo [ERROR] Failed to install Hunyuan3D 2.0 package
    exit /b 1
)

echo [INFO] Building custom rasterizer...
cd hy3dgen\texgen\custom_rasterizer
call python setup.py install
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Custom rasterizer built successfully
) else (
    echo [ERROR] Failed to build custom rasterizer
    exit /b 1
)

echo [INFO] Building differentiable renderer...
cd ..\..\..
cd hy3dgen\texgen\differentiable_renderer
call python setup.py install
cd ..\..\..\
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Differentiable renderer built successfully
) else (
    echo [ERROR] Failed to build differentiable renderer
    exit /b 1
)
REM installation for hunyuan3d 2.0 end
echo [SUCCESS] Hunyuan3D 2.0 installation completed

echo.
echo ========================================
echo Installing Hunyuan3D 2.1 Dependencies
echo ========================================
REM installation for hunyuan3d 2.1
echo [INFO] Changing directory to thirdparty\Hunyuan3D-2.1...
cd ..\..\thirdparty\Hunyuan3D-2.1
echo [INFO] Installing custom rasterizer for Hunyuan3D 2.1...
@REM cd hy3dpaint\custom_rasterizer
@REM call pip install -e .
pip install https://github.com/Deathdadev/Hunyuan3D-2.1/releases/download/windows-whl/custom_rasterizer-0.1-cp310-cp310-win_amd64.whl
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Hunyuan3D 2.1 custom rasterizer installed
) else (
    echo [ERROR] Failed to install Hunyuan3D 2.1 custom rasterizer
    exit /b 1
)

echo [INFO] Building differentiable renderer for Hunyuan3D 2.1...
@REM On windows we download a prebuilt pyd, its much more convenient 
wget https://github.com/Deathdadev/Hunyuan3D-2.1/releases/download/windows-whl/mesh_inpaint_processor.cp310-win_amd64.pyd -O hy3dpaint\DifferentiableRenderer\mesh_inpaint_processor.cp310-win_amd64.pyd
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Hunyuan3D 2.1 differentiable renderer built successfully
) else (
    echo [ERROR] Failed to build Hunyuan3D 2.1 differentiable renderer
    exit /b 1
)
echo [INFO] Installing Hunyuan3D 2.1 requirements...
call pip install -r requirements-inference.txt 
REM installation for hunyuan3d 2.1 end
echo [SUCCESS] Hunyuan3D 2.1 installation completed

echo.
echo ========================================
echo Installing HoloPart Dependencies
echo ========================================
REM holopart for part completion
echo [INFO] Changing directory to thirdparty\HoloPart...
cd ..\..\thirdparty\HoloPart
echo [INFO] Installing HoloPart requirements...
call pip install -r requirements.txt 
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] HoloPart requirements installed
) else (
    echo [ERROR] Failed to install HoloPart requirements
    exit /b 1
)
REM holopart for part completion end
echo [SUCCESS] HoloPart installation completed

echo.
echo ========================================
echo Installing UniRig Dependencies
echo ========================================
REM unirig for auto-rigging
echo [INFO] Changing directory to thirdparty\UniRig...
cd ..\..\thirdparty\UniRig
echo [INFO] Installing spconv-cu120 for UniRig...
call pip install spconv-cu120
call pip install pyrender fast-simplification python-box timm
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] UniRig dependencies installed
) else (
    echo [ERROR] Failed to install UniRig dependencies
    exit /b 1
)

echo.
echo ========================================
echo Installing PartPacker Dependencies
echo ========================================
REM part packer
echo [INFO] Changing directory to thirdparty\PartPacker...
cd ..\..\thirdparty\PartPacker
echo [INFO] Installing PartPacker requirements...
call pip install meshiki fpsample kiui pymcubes einops
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] PartPacker requirements installed
) else (
    echo [ERROR] Failed to install PartPacker requirements
    exit /b 1
)
REM part packer end
echo [SUCCESS] PartPacker installation completed

cd ..\..

echo.
echo ========================================
echo Installing Project Dependencies
echo ========================================
REM for this project (fastapi / uvicorn relevant etc.)
echo [INFO] Installing main project requirements...
call pip install -r requirements.txt 
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Main project requirements installed
) else (
    echo [ERROR] Failed to install main project requirements
    exit /b 1
)

echo [INFO] Installing test requirements...
REM testing 
call pip install -r requirements-test.txt 
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Test requirements installed
) else (
    echo [ERROR] Failed to install test requirements
    exit /b 1
)

echo [INFO] Installing huggingface_hub for model downloading...
REM for downloading models 
call pip install huggingface_hub
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] huggingface_hub installed
) else (
    echo [ERROR] Failed to install huggingface_hub
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo All installation done successfully!

echo Checking CUDA availability...
python -c "import torch; print(torch.cuda.is_available())" && echo CUDA installed successfully || echo Failed

echo Checking PyTorch version...
python -c "import torch; print(torch.__version__)" && echo PyTorch installed successfully || echo Failed

echo Checking Blender availability...
python -c "import bpy" && echo Blender installed successfully || echo Failed

echo Checking Other Packages...
python -c "import kaolin; print(kaolin.__version__)" && echo Kaolin installed successfully || echo Failed
python -c "import open3d; import pymeshlab" && echo Open3D and pymeshlab installed successfully || echo Failed

echo [NOTE] Windows does not require libsm6 installation like Linux systems
echo [INFO] Installation completed successfully on Windows!

endlocal 