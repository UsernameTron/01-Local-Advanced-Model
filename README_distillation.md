# Executor Knowledge Distillation Pipeline

This pipeline distills knowledge from a large teacher model (e.g., Phi3.5 or Phi-2) to a smaller student model (e.g., TinyLlama) for use as a fast executor agent in the Local O1 system.

## Usage Instructions

### 1. Install Requirements
```zsh
pip install torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install transformers datasets
```

### 2. Prepare Data
- Place your training and validation data in `dataset/executor_distillation_train.json` and `dataset/executor_distillation_val.json`.
- Adjust `distill_config.json` as needed for your models and hyperparameters.

### 3. Run Distillation
```zsh
python distill_executor.py
```
- Training will checkpoint and log metrics to `distill_output/`.
- Early stopping is enabled by default.

### 4. Export and Import Model
- The distilled model and tokenizer are saved to `executor-distilled/`.
- To import into Ollama:
```zsh
ollama import executor-distilled:latest ./executor-distilled
```

## Expected Outcomes
- A student model under 2GB, compatible with Ollama and Apple MPS.
- Training metrics in `distill_output/training_metrics.json`.
- Loss curves and validation scores for monitoring.

## Troubleshooting
- If you encounter MPS memory errors, reduce `batch_size` or increase `grad_accum_steps` in `distill_config.json`.
- For recovery, resume from the latest checkpoint in `distill_output/`.

## Hyperparameters
See `distill_config.json` for all settings (learning rate, batch size, epochs, temperature, etc).

---

**This pipeline is designed for reproducibility, efficient Apple MPS training, and seamless Ollama integration.**
