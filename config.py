CEO_MODEL = 'phi3.5'
FAST_MODEL = 'phi3.5'
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.95

# Quantization and mixed precision inference settings for M4 Pro
# Use F16 precision and optimized KV cache for Apple MPS backend
MODEL_PRECISION = 'F16'  # For Ollama model quantization
MPS_OPTIMIZED = True
KV_CACHE_TYPE = 'f16'  # Optimized for Apple Metal
MAX_EXECUTOR_STEPS = 5  # Limit executor steps for performance

# Executor model selection
EXECUTOR_MODEL_ORIGINAL = 'phi3.5'
EXECUTOR_MODEL_DISTILLED = 'executor-distilled'
# Default: use distilled, can be switched in agent factory
USE_DISTILLED_EXECUTOR = True

# Metal/MPS-specific environment variables (to be set in shell or scripts)
# export OLLAMA_METAL=1
# export OLLAMA_MPS=1
# export OLLAMA_KV_CACHE_TYPE=f16
# export OLLAMA_MODEL_PRECISION=F16

# Memory manager placeholder (to be implemented in utils.py)
from utils import OptimizedMemoryManager
memory_manager = OptimizedMemoryManager(max_ram_gb=48)
memory_manager.enable_auto_compression()
