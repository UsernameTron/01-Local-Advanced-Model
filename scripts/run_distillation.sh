#!/bin/zsh
# One-command execution for the full distillation pipeline

set -e

# Activate your virtual environment if needed
# source venv/bin/activate

# Install requirements (uncomment if needed)
# pip install torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu
# pip install transformers datasets

# Set Apple MPS environment variables
export OLLAMA_MPS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run distillation
python distill_executor.py

# Import model to Ollama (uncomment if ready)
# ollama import executor-distilled:latest ./executor-distilled
