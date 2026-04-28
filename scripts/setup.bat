@echo off
setlocal
REM One-time setup for the DeepSeek RAG project.
REM Run from the project root (deepseek-rag\), not from inside scripts\.

pushd "%~dp0.."

echo ============================================================
echo   DeepSeek RAG - One-Time Setup
echo ============================================================
echo.

if not exist .env (
    echo [!] .env not found. Copying from .env.example.
    copy /Y .env.example .env >nul
    echo     Edit .env and fill in DEEPSEEK_API_KEY and REPO_PATH before continuing.
    popd
    exit /b 1
)

if not exist venv (
    echo [1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [!] Failed to create venv. Install Python 3.11 and retry.
        popd
        exit /b 1
    )
) else (
    echo [1/5] venv already exists.
)

call venv\Scripts\activate

echo.
echo [2/5] Installing PyTorch with CUDA 11.8 (this downloads ~2.5GB)...
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [!] PyTorch install failed.
    popd
    exit /b 1
)

echo.
echo [3/5] Installing remaining dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [!] pip install failed.
    popd
    exit /b 1
)

echo.
echo [4/5] GPU diagnostics...
python -m src.utils.gpu_check

echo.
echo [5/5] Initial repo indexing (takes 3-8 minutes the first time)...
python -m src.indexer

echo.
echo ============================================================
echo   Setup complete. Run scripts\start.bat to launch the proxy.
echo ============================================================
popd
endlocal
pause
