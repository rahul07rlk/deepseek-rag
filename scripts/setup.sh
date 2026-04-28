#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  DeepSeek RAG - One-Time Setup"
echo "============================================================"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "[!] .env created from .env.example. Fill in DEEPSEEK_API_KEY and REPO_PATH, then re-run."
    exit 1
fi

if [ ! -d venv ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/5] venv already exists."
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "[2/5] Installing PyTorch with CUDA 11.8..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

echo "[3/5] Installing remaining dependencies..."
pip install -r requirements.txt

echo "[4/5] GPU diagnostics..."
python -m src.utils.gpu_check

echo "[5/5] Initial repo indexing..."
python -m src.indexer

echo
echo "Setup complete. Run scripts/start.sh to launch the proxy."
