@echo off
echo Installing Genesis OS...

:: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment!
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment!
    exit /b 1
)

:: Check for CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected. Installing CPU version...
    pip install -e .
) else (
    echo NVIDIA GPU detected. Installing with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -e .
)

echo.
echo Installation complete! You can now run Genesis OS using:
echo python run_genesis.py
echo.
pause 