#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source venv/bin/activate
echo "Force re-indexing entire repo..."
python -c "from src.indexer import index_repo; index_repo(force=True)"
