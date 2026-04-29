@echo off
setlocal
pushd "%~dp0"

call venv\Scripts\activate

echo ============================================================
echo   Qwen3 Rerank Server - Starting
echo ============================================================
echo.
echo   Listening on  : http://0.0.0.0:9000
echo   Health check  : http://localhost:9000/health
echo   Endpoint      : POST /rerank
echo.
echo   On the MAIN laptop set in deepseek-rag\.env:
echo     RERANKER_PROVIDER=remote
echo     REMOTE_RERANKER_URL=http://^<this-laptop-LAN-IP^>:9000
echo.
echo   Press Ctrl+C to stop.
echo ============================================================
echo.

python server.py

popd
endlocal
