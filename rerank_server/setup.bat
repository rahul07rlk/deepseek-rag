@echo off
setlocal
REM One-time setup for the standalone rerank server (second laptop).
REM Run this once on the SECOND laptop after copying the rerank_server\ folder.

pushd "%~dp0"

echo ============================================================
echo   Qwen3 Rerank Server - Setup
echo ============================================================
echo.

if not exist venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [!] venv creation failed. Install Python 3.11+ and retry.
        popd
        exit /b 1
    )
) else (
    echo [1/3] venv already exists.
)

call venv\Scripts\activate

echo.
echo [2/3] Installing PyTorch with CUDA 11.8 (downloads ~2.5GB)...
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [!] PyTorch install failed.
    popd
    exit /b 1
)

echo.
echo [3/3] Installing remaining dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [!] pip install failed.
    popd
    exit /b 1
)

echo.
echo ============================================================
echo   Setup complete.
echo.
echo   Next steps:
echo     1. Find this laptop's LAN IP:    ipconfig
echo        (look for IPv4 Address under your Wi-Fi adapter)
echo     2. Allow port 9000 through Windows Firewall (Private network).
echo     3. Run start.bat to launch the server.
echo ============================================================
popd
endlocal
pause
