@echo off
setlocal enabledelayedexpansion

REM 3DAIGC Model Download Script
REM Usage: download_models.bat [OPTIONS]
REM 
REM Available models:
REM   partfield, hunyuan2, hunyuan2mini, hunyuan21, trellis, trellis-text, 
REM   holopart, unirig, partpacker, misc, all
REM
REM Options:
REM   -h, --help              Show this help message
REM   -m, --models MODEL      Comma-separated list of models to download (default: all)
REM   -v, --verify            Verify existing models without downloading
REM   -f, --force             Force re-download even if files exist
REM   --list                  List all available models

REM Default values
set MODELS_TO_DOWNLOAD=all
set VERIFY_ONLY=false
set FORCE_DOWNLOAD=false

REM Print functions (using echo since Windows batch doesn't have colored output natively)
goto :parse_args

:print_info
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:show_help
echo 3DAIGC Model Download Script
echo.
echo Usage: %~n0 [OPTIONS]
echo.
echo Options:
echo     -h, --help              Show this help message
echo     -m, --models MODELS     Comma-separated list of models to download (default: all)
echo     -v, --verify            Verify existing models without downloading
echo     -f, --force             Force re-download even if files exist
echo     --list                  List all available models
echo.
echo Available models:
echo     partfield     - PartField model for mesh segmentation
echo     hunyuan2      - Hunyuan3D 2.0 models (geometry/texture/vae)
echo     hunyuan2mini  - Hunyuan3D 2.0 mini models
echo     hunyuan21     - Hunyuan3D 2.1 models  
echo     trellis       - TRELLIS image-large model
echo     trellis-text  - TRELLIS text-xlarge model (optional)
echo     holopart      - HoloPart model for part completion
echo     unirig        - UniRig model for auto-rigging
echo     partpacker    - PartPacker model
echo     misc          - Miscellaneous models (RealESRGAN, DINOv2)
echo     all           - Download all models
echo.
echo Examples:
echo     %~n0                                    # Download all models
echo     %~n0 -m hunyuan2,trellis               # Download only Hunyuan3D 2.0 and TRELLIS
echo     %~n0 -v                                # Verify all existing models
echo     %~n0 -m partfield -f                   # Force re-download PartField model
echo     %~n0 --list                           # List available models
echo.
goto :eof

:list_models
echo Available models:
echo   - partfield
echo   - hunyuan2
echo   - hunyuan2mini
echo   - hunyuan21
echo   - trellis
echo   - trellis-text
echo   - holopart
echo   - unirig
echo   - partpacker
echo   - misc
goto :eof

:parse_args
if "%~1"=="" goto :main_execution
if /I "%~1"=="-h" goto :show_help_and_exit
if /I "%~1"=="--help" goto :show_help_and_exit
if /I "%~1"=="-m" (
    set MODELS_TO_DOWNLOAD=%~2
    shift
    shift
    goto :parse_args
)
if /I "%~1"=="--models" (
    set MODELS_TO_DOWNLOAD=%~2
    shift
    shift
    goto :parse_args
)
if /I "%~1"=="-v" (
    set VERIFY_ONLY=true
    shift
    goto :parse_args
)
if /I "%~1"=="--verify" (
    set VERIFY_ONLY=true
    shift
    goto :parse_args
)
if /I "%~1"=="-f" (
    set FORCE_DOWNLOAD=true
    shift
    goto :parse_args
)
if /I "%~1"=="--force" (
    set FORCE_DOWNLOAD=true
    shift
    goto :parse_args
)
if /I "%~1"=="--list" goto :list_models_and_exit
call :print_error "Unknown option: %~1"
goto :show_help_and_exit

:show_help_and_exit
call :show_help
exit /b 0

:list_models_and_exit
call :list_models
exit /b 0

:verify_file
set file_path=%~1
set min_size=%~2
if "%min_size%"=="" set min_size=1000

if exist "%file_path%" (
    for %%I in ("%file_path%") do set file_size=%%~zI
    if !file_size! gtr %min_size% (
        call :print_success "✓ %file_path% (!file_size! bytes)"
        exit /b 0
    ) else (
        call :print_warning "✗ %file_path% exists but is too small (!file_size! bytes)"
        exit /b 1
    )
) else (
    call :print_warning "✗ %file_path% not found"
    exit /b 1
)

:verify_directory
set dir_path=%~1
set min_files=%~2
if "%min_files%"=="" set min_files=1

if exist "%dir_path%" (
    set file_count=0
    for /r "%dir_path%" %%f in (*) do set /a file_count+=1
    if !file_count! geq %min_files% (
        call :print_success "✓ %dir_path% (!file_count! files)"
        exit /b 0
    ) else (
        call :print_warning "✗ %dir_path% exists but has insufficient files (!file_count! files, need %min_files%)"
        exit /b 1
    )
) else (
    call :print_warning "✗ %dir_path% not found"
    exit /b 1
)

:download_with_verify
set url=%~1
set output_path=%~2
set description=%~3

call :print_info "Downloading %description%..."
call :print_info "URL: %url%"
call :print_info "Output: %output_path%"

REM Create directory if it doesn't exist
for %%f in ("%output_path%") do md "%%~dpf" 2>nul

REM Download the file using PowerShell (since wget is not native on Windows)
powershell -Command "Invoke-WebRequest -Uri '%url%' -OutFile '%output_path%'"
if %ERRORLEVEL% equ 0 (
    call :verify_file "%output_path%"
    if !ERRORLEVEL! equ 0 (
        call :print_success "Successfully downloaded %description%"
        exit /b 0
    ) else (
        call :print_error "Downloaded file verification failed for %description%"
        exit /b 1
    )
) else (
    call :print_error "Failed to download %description%"
    exit /b 1
)

:download_partfield
call :print_info "========================================"
call :print_info "Downloading PartField Model"
call :print_info "========================================"

set model_path=pretrained\PartField\model_objaverse.pt

if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_file "%model_path%" 50000000
    if !ERRORLEVEL! equ 0 (
        call :print_info "PartField model already exists and verified"
        exit /b 0
    )
)

md pretrained\PartField 2>nul
call :download_with_verify "https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt" "%model_path%" "PartField model"
exit /b %ERRORLEVEL%

:download_hunyuan2
call :print_info "========================================"
call :print_info "Downloading Hunyuan3D 2.0 Models"
call :print_info "========================================"

set model_dir=pretrained\tencent\Hunyuan3D-2

if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_directory "%model_dir%" 10
    if !ERRORLEVEL! equ 0 (
        call :print_info "Hunyuan3D 2.0 models already exist and verified"
        exit /b 0
    )
)

md "%model_dir%" 2>nul
call :print_info "Downloading Hunyuan3D 2.0 (geometry/texture/vae)..."
call :print_info "Notice that the paint and delight models are ONLY needed when you need to texture generation feature"
call huggingface-cli download tencent/Hunyuan3D-2 --include "hunyuan3d-dit-v2-0-turbo/*" "hunyuan3d-vae-v2-0-turbo/*" "hunyuan3d-paint-v2-0-turbo/*" "hunyuan3d-delight-v2-0/*" --local-dir "%model_dir%"
if %ERRORLEVEL% equ 0 (
    call :print_success "Hunyuan3D 2.0 models downloaded successfully"
) else (
    call :print_error "Failed to download Hunyuan3D 2.0 models"
    exit /b 1
)
exit /b 0

:download_hunyuan2mini
call :print_info "========================================"
call :print_info "Downloading Hunyuan3D 2.0 Mini Models"
call :print_info "========================================"

set model_dir=pretrained\tencent\Hunyuan3D-2mini

if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_directory "%model_dir%" 5
    if !ERRORLEVEL! equ 0 (
        call :print_info "Hunyuan3D 2.0 mini models already exist and verified"
        exit /b 0
    )
)

md "%model_dir%" 2>nul
call :print_info "Downloading Hunyuan3D 2.0 mini (geometry/vae)..."
call huggingface-cli download tencent/Hunyuan3D-2mini --include "hunyuan3d-dit-v2-mini-turbo/*" "hunyuan3d-vae-v2-mini-turbo/*" --local-dir "%model_dir%"
if %ERRORLEVEL% equ 0 (
    call :print_success "Hunyuan3D 2.0 mini models downloaded successfully"
) else (
    call :print_error "Failed to download Hunyuan3D 2.0 mini models"
    exit /b 1
)
exit /b 0

:download_hunyuan21
call :print_info "========================================"
call :print_info "Downloading Hunyuan3D 2.1 Models"
call :print_info "========================================"

set model_dir=pretrained\tencent\Hunyuan3D-2.1

if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_directory "%model_dir%" 5
    if !ERRORLEVEL! equ 0 (
        call :print_info "Hunyuan3D 2.1 models already exist and verified"
        exit /b 0
    )
)

md "%model_dir%" 2>nul
call :print_info "Downloading Hunyuan3D 2.1 models..."
call huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir "%model_dir%"
if %ERRORLEVEL% equ 0 (
    call :print_success "Hunyuan3D 2.1 models downloaded successfully"
) else (
    call :print_error "Failed to download Hunyuan3D 2.1 models"
    exit /b 1
)
exit /b 0

:download_trellis
call :print_info "========================================"
call :print_info "Downloading TRELLIS Image-Large Model"
call :print_info "========================================"

set model_dir=pretrained\TRELLIS\TRELLIS-image-large

if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_directory "%model_dir%" 5
    if !ERRORLEVEL! equ 0 (
        call :print_info "TRELLIS image-large model already exists and verified"
        exit /b 0
    )
)

md "%model_dir%" 2>nul
call :print_info "Downloading TRELLIS image-large model..."
call huggingface-cli download microsoft/TRELLIS-image-large --local-dir "%model_dir%"
if %ERRORLEVEL% equ 0 (
    call :print_success "TRELLIS image-large model downloaded successfully"
) else (
    call :print_error "Failed to download TRELLIS image-large model"
    exit /b 1
)
exit /b 0

:download_trellis_text
call :print_info "========================================"
call :print_info "Downloading TRELLIS Text-XLarge Model"
call :print_info "========================================"

set model_dir=pretrained\TRELLIS\TRELLIS-text-xlarge

if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_directory "%model_dir%" 5
    if !ERRORLEVEL! equ 0 (
        call :print_info "TRELLIS text-xlarge model already exists and verified"
        exit /b 0
    )
)

md "%model_dir%" 2>nul
call :print_info "Downloading TRELLIS text-xlarge model (optional, for text-conditioned part re-texturing)..."
call huggingface-cli download microsoft/TRELLIS-text-xlarge --local-dir "%model_dir%"
if %ERRORLEVEL% equ 0 (
    call :print_success "TRELLIS text-xlarge model downloaded successfully"
) else (
    call :print_error "Failed to download TRELLIS text-xlarge model"
    exit /b 1
)
exit /b 0

:download_holopart
call :print_info "========================================"
call :print_info "Downloading HoloPart Model"
call :print_info "========================================"

set model_dir=pretrained\HoloPart

if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_directory "%model_dir%" 3
    if !ERRORLEVEL! equ 0 (
        call :print_info "HoloPart model already exists and verified"
        exit /b 0
    )
)

md "%model_dir%" 2>nul
call :print_info "Downloading HoloPart model..."
call huggingface-cli download VAST-AI/HoloPart --local-dir "%model_dir%"
if %ERRORLEVEL% equ 0 (
    call :print_success "HoloPart model downloaded successfully"
) else (
    call :print_error "Failed to download HoloPart model"
    exit /b 1
)
exit /b 0

:download_unirig
call :print_info "========================================"
call :print_info "Downloading UniRig Model"
call :print_info "========================================"

set model_dir=pretrained\UniRig

if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_directory "%model_dir%" 3
    if !ERRORLEVEL! equ 0 (
        call :print_info "UniRig model already exists and verified"
        exit /b 0
    )
)

md "%model_dir%" 2>nul
call :print_info "Downloading UniRig model..."
call huggingface-cli download VAST-AI/UniRig --local-dir "%model_dir%"
if %ERRORLEVEL% equ 0 (
    call :print_success "UniRig model downloaded successfully"
) else (
    call :print_error "Failed to download UniRig model"
    exit /b 1
)
exit /b 0

:download_partpacker
call :print_info "========================================"
call :print_info "Downloading PartPacker Model"
call :print_info "========================================"

set model_dir=pretrained\PartPacker

if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_directory "%model_dir%" 3
    if !ERRORLEVEL! equ 0 (
        call :print_info "PartPacker model already exists and verified"
        exit /b 0
    )
)

md "%model_dir%" 2>nul
call :print_info "Downloading PartPacker model..."
call huggingface-cli download nvidia/PartPacker --local-dir "%model_dir%"
if %ERRORLEVEL% equ 0 (
    call :print_success "PartPacker model downloaded successfully"
) else (
    call :print_error "Failed to download PartPacker model"
    exit /b 1
)
exit /b 0

:download_misc
call :print_info "========================================"
call :print_info "Downloading Miscellaneous Models"
call :print_info "========================================"

REM RealESRGAN_x4plus for Hunyuan3D-2.1
set realesrgan_path=pretrained\misc\RealESRGAN_x4plus.pth
if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_file "%realesrgan_path%" 50000000
    if !ERRORLEVEL! equ 0 (
        call :print_info "RealESRGAN_x4plus already exists and verified"
        goto :download_misc_dinov2
    )
)

md pretrained\misc 2>nul
call :download_with_verify "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" "%realesrgan_path%" "RealESRGAN_x4plus model"

:download_misc_dinov2
REM DINOv2-giant used in PartPacker or elsewhere
set dinov2_dir=pretrained\dinov2-giant
if /I "%FORCE_DOWNLOAD%"=="false" (
    call :verify_directory "%dinov2_dir%" 5
    if !ERRORLEVEL! equ 0 (
        call :print_info "DINOv2-giant model already exists and verified"
        exit /b 0
    )
)

md "%dinov2_dir%" 2>nul
call :print_info "Downloading DINOv2-giant model..."
call huggingface-cli download facebook/dinov2-giant --local-dir "%dinov2_dir%" --exclude "*.bin"
if %ERRORLEVEL% equ 0 (
    call :print_success "DINOv2-giant model downloaded successfully"
) else (
    call :print_error "Failed to download DINOv2-giant model"
    exit /b 1
)
exit /b 0

:verify_all_models
call :print_info "========================================"
call :print_info "Verifying All Models"
call :print_info "========================================"

set all_verified=true

call :print_info "Checking PartField..."
call :verify_file "pretrained\PartField\model_objaverse.pt" 50000000
if !ERRORLEVEL! neq 0 set all_verified=false

call :print_info "Checking Hunyuan3D 2.0..."
call :verify_directory "pretrained\tencent\Hunyuan3D-2" 10
if !ERRORLEVEL! neq 0 set all_verified=false

call :print_info "Checking Hunyuan3D 2.0 mini..."
call :verify_directory "pretrained\tencent\Hunyuan3D-2mini" 5
if !ERRORLEVEL! neq 0 set all_verified=false

call :print_info "Checking Hunyuan3D 2.1..."
call :verify_directory "pretrained\tencent\Hunyuan3D-2.1" 5
if !ERRORLEVEL! neq 0 set all_verified=false

call :print_info "Checking TRELLIS image-large..."
call :verify_directory "pretrained\TRELLIS\TRELLIS-image-large" 5
if !ERRORLEVEL! neq 0 set all_verified=false

call :print_info "Checking TRELLIS text-xlarge (optional)..."
call :verify_directory "pretrained\TRELLIS\TRELLIS-text-xlarge" 5
if !ERRORLEVEL! neq 0 call :print_warning "TRELLIS text-xlarge not found (optional)"

call :print_info "Checking HoloPart..."
call :verify_directory "pretrained\HoloPart" 3
if !ERRORLEVEL! neq 0 set all_verified=false

call :print_info "Checking UniRig..."
call :verify_directory "pretrained\UniRig" 3
if !ERRORLEVEL! neq 0 set all_verified=false

call :print_info "Checking PartPacker..."
call :verify_directory "pretrained\PartPacker" 3
if !ERRORLEVEL! neq 0 set all_verified=false

call :print_info "Checking miscellaneous models..."
call :verify_file "pretrained\misc\RealESRGAN_x4plus.pth" 50000000
if !ERRORLEVEL! neq 0 set all_verified=false
call :verify_directory "pretrained\dinov2-giant" 5
if !ERRORLEVEL! neq 0 set all_verified=false

if /I "%all_verified%"=="true" (
    call :print_success "All required models are present and verified!"
) else (
    call :print_warning "Some models are missing or corrupted. Run without -v flag to download them."
)
exit /b 0

:main_execution
call :print_info "========================================"
call :print_info "3DAIGC Model Download Script"
call :print_info "========================================"

REM Check if huggingface-cli is available
where huggingface-cli >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "huggingface-cli is not installed. Please install it first:"
    call :print_error "pip install huggingface_hub"
    exit /b 1
)

REM Check if PowerShell is available (for downloading files)
where powershell >nul 2>&1
if %ERRORLEVEL% neq 0 (
    call :print_error "PowerShell is not available. This script requires PowerShell for file downloads."
    exit /b 1
)

REM If verify only mode
if /I "%VERIFY_ONLY%"=="true" (
    call :verify_all_models
    exit /b 0
)

REM Download requested models
set models_list=%MODELS_TO_DOWNLOAD%
set models_list=!models_list:,= !

for %%m in (%models_list%) do (
    if /I "%%m"=="partfield" call :download_partfield
    if /I "%%m"=="hunyuan2" call :download_hunyuan2
    if /I "%%m"=="hunyuan2mini" call :download_hunyuan2mini
    if /I "%%m"=="hunyuan21" call :download_hunyuan21
    if /I "%%m"=="trellis" call :download_trellis
    if /I "%%m"=="trellis-text" call :download_trellis_text
    if /I "%%m"=="holopart" call :download_holopart
    if /I "%%m"=="unirig" call :download_unirig
    if /I "%%m"=="partpacker" call :download_partpacker
    if /I "%%m"=="misc" call :download_misc
    if /I "%%m"=="all" (
        call :download_partfield
        call :download_hunyuan2
        call :download_hunyuan2mini
        call :download_hunyuan21
        call :download_trellis
        call :download_trellis_text
        call :download_holopart
        call :download_unirig
        call :download_partpacker
        call :download_misc
    )
)

call :print_success "========================================"
call :print_success "Model Download Complete!"
call :print_success "========================================"
call :print_info "All requested models have been downloaded successfully."
call :print_info "You can verify the downloads by running: %~n0 -v"

endlocal 