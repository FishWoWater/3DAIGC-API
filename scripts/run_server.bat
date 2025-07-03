@echo off
setlocal enabledelayedexpansion

echo 🚀 Starting 3D Generative Models Backend...

REM Check if configuration files exist
if not exist "config\system.yaml" (
    echo ❌ Configuration file config\system.yaml not found
    echo Please run .\scripts\setup.bat to create configuration files
    exit /b 1
)

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Default values
if "%P3D_HOST%"=="" set P3D_HOST=0.0.0.0
if "%P3D_PORT%"=="" set P3D_PORT=7842
if "%P3D_WORKERS%"=="" set P3D_WORKERS=1
if "%P3D_RELOAD%"=="" set P3D_RELOAD=false
if "%P3D_LOG_LEVEL%"=="" set P3D_LOG_LEVEL=info

echo 📋 Configuration:
echo    Host: %P3D_HOST%
echo    Port: %P3D_PORT%
echo    Workers: %P3D_WORKERS%
echo    Reload: %P3D_RELOAD%
echo    Log Level: %P3D_LOG_LEVEL%
echo.

REM Start server based on environment
if /I "%P3D_RELOAD%"=="true" (
    echo 🔄 Starting development server with auto-reload...
    call uvicorn api.main:app --host %P3D_HOST% --port %P3D_PORT% --reload --log-level %P3D_LOG_LEVEL%
) else (
    echo 🚀 Starting production server...
    if "%P3D_WORKERS%"=="1" (
        REM Single worker
        call uvicorn api.main:app --host %P3D_HOST% --port %P3D_PORT% --log-level %P3D_LOG_LEVEL%
    ) else (
        REM Multiple workers with Gunicorn (Note: Gunicorn doesn't work on Windows, using uvicorn instead)
        echo [WARNING] Multiple workers with Gunicorn is not supported on Windows
        echo [INFO] Starting single worker uvicorn server instead
        call uvicorn api.main:app --host %P3D_HOST% --port %P3D_PORT% --log-level %P3D_LOG_LEVEL%
    )
)

endlocal 