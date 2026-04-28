@echo off
setlocal
pushd "%~dp0.."

echo ============================================================
echo   DeepSeek RAG Proxy - Starting
echo ============================================================

call venv\Scripts\activate

echo [1/2] Starting file watcher in a separate window...
start "RAG File Watcher" cmd /k "call venv\Scripts\activate && python -m src.watcher"

echo [2/2] Starting proxy server...
echo.
echo   Proxy URL : http://localhost:8000
echo   Status    : http://localhost:8000/
echo   Stats     : http://localhost:8000/stats
echo.
echo   Point your IDE to: http://localhost:8000/v1
echo   Press Ctrl+C to stop.
echo.
python -m src.proxy_server

popd
endlocal
