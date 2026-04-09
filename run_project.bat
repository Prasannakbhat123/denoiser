@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"


set "MODE=%~1"
if "%MODE%"=="" set "MODE=app"

set "PY_CMD=python"
where py >nul 2>nul
if %errorlevel%==0 set "PY_CMD=py -3"

if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv.
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo Upgrading pip and installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

set "BASE_REQ=.venv\requirements_base.txt"
python -c "from pathlib import Path; req = Path('requirements.txt').read_text(encoding='utf-8').splitlines(); keep = [line for line in req if line.strip() and not line.lower().startswith(('torch', 'torchvision'))]; Path(r'%BASE_REQ%').write_text('\n'.join(keep) + '\n', encoding='utf-8')"
if errorlevel 1 exit /b 1

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
    echo NVIDIA GPU detected. Installing CUDA-enabled PyTorch wheels...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo No NVIDIA GPU detected. Installing CPU PyTorch wheels...
    python -m pip install torch torchvision
)
if errorlevel 1 exit /b 1

python -m pip install -r "%BASE_REQ%"
if errorlevel 1 exit /b 1

if /I "%MODE%"=="benchmark" (
    echo Running benchmark across multiple input sizes...
    python benchmark.py --sizes 64,128,256 --samples 8 --repeats 10 --warmup 3 --output-dir artifacts_benchmark
) else if /I "%MODE%"=="train" (
    echo Training model using paired clean and noisy images...
    if not exist "data_noisy" mkdir data_noisy
    python train_model.py --clean-dir data --noisy-dir data_noisy --make-noisy-first --noise-level 30 --epochs 20 --batch-size 8 --image-size 128 --output-dir artifacts_paired
) else (
    echo Starting Streamlit app...
    python -m streamlit run denoisingapp.py --server.headless false
)

if exist "%BASE_REQ%" del "%BASE_REQ%" >nul 2>nul

endlocal