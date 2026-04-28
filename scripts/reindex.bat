@echo off
setlocal
pushd "%~dp0.."
call venv\Scripts\activate
echo Force re-indexing entire repo...
python -c "from src.indexer import index_repo; index_repo(force=True)"
popd
endlocal
pause
