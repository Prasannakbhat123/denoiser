@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [ERROR] Virtual environment not found at .venv.
    echo Create it first with: python -m venv .venv
    exit /b 1
)

echo Starting Streamlit app...
python -m streamlit run denoisingapp.py --server.headless false

endlocal