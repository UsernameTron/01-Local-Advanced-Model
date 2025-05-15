# Executor Distillation Dataset

This dataset is designed for knowledge distillation of a lightweight executor agent model for the Local O1 system.

## Structure
- `executor_distillation_train.json`: Training set (input prompt, target response pairs)
- `executor_distillation_val.json`: Validation set (input prompt, target response pairs)

## Statistics
- **Training examples:** 10
- **Validation examples:** 2
- **Max response length:** < 100 tokens
- **Domains:** Planning, code generation, dependency analysis, summarization

## Sample Example
```
{"input": "Step 4: Generate pytest tests for agents.py.", "target": "Write tests for Agent class methods and edge cases."}
```

## Loading Script (HuggingFace Datasets)
```python
from datasets import load_dataset
train_ds = load_dataset('json', data_files='dataset/executor_distillation_train.json')['train']
val_ds = load_dataset('json', data_files='dataset/executor_distillation_val.json')['train']
```
