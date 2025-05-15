#!/bin/zsh
set -e

# 1. Python & venv
if ! command -v python3 &>/dev/null; then
  echo "Python3 not found. Install Python 3.10+ first."; exit 1
fi
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 2. Core dependencies
pip install -r requirements.txt || pip install streamlit psutil faiss-cpu torch transformers sentence-transformers requests

# 3. Ollama
if ! command -v ollama &>/dev/null; then
  echo "Ollama not found. Installing..."
  if [[ "$(uname)" == "Darwin" ]]; then
    brew install ollama
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
fi
ollama serve &
ollama pull phi3.5

# 4. Create default benchmark file if missing
python scripts/create_benchmark_results.py

# 5. Permissions
chmod +x scripts/*.sh

# 6. (Optional) Docker build
# docker build -t local-o1 .

echo "[OK] Local O1 setup complete."
