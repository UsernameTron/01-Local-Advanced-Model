import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
from torch.nn import functional as F

# Load config
with open('distill_config.json') as f:
    config = json.load(f)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OLLAMA_MPS'] = '1'
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def get_teacher_logits(teacher, tokenizer, inputs, temperature):
    with torch.no_grad():
        inputs_enc = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=config['max_length']).to(device)
        outputs = teacher(**inputs_enc)
        logits = outputs.logits / temperature
        return F.log_softmax(logits, dim=-1)

def distillation_loss(student_logits, teacher_logits, labels, temperature):
    # Student loss: cross-entropy with teacher's soft targets
    loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    return loss

class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        outputs = model(input_ids=input_ids, labels=labels)
        student_logits = outputs.logits
        # Get teacher logits
        teacher_logits = get_teacher_logits(self.teacher, self.tokenizer, inputs['input_text'], self.args.temperature)
        loss = distillation_loss(student_logits, teacher_logits, labels, self.args.temperature)
        return (loss, outputs) if return_outputs else loss

# Load models and tokenizer
teacher = AutoModelForCausalLM.from_pretrained(config['teacher_model'], torch_dtype=torch.float16, device_map='mps')
student = AutoModelForCausalLM.from_pretrained(config['student_model'], torch_dtype=torch.float16, device_map='mps')
tokenizer = AutoTokenizer.from_pretrained(config['student_model'])

# Load dataset
train_ds = load_dataset('json', data_files=config['train_file'])['train']
val_ds = load_dataset('json', data_files=config['val_file'])['train']

# Preprocess dataset
max_length = config['max_length']
def preprocess(example):
    enc = tokenizer(example['input'], truncation=True, padding='max_length', max_length=max_length)
    enc['labels'] = tokenizer(example['target'], truncation=True, padding='max_length', max_length=max_length)['input_ids']
    enc['input_text'] = example['input']
    return enc

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# Training arguments
training_args = TrainingArguments(
    output_dir=config['output_dir'],
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'],
    gradient_accumulation_steps=config['grad_accum_steps'],
    num_train_epochs=config['epochs'],
    learning_rate=config['learning_rate'],
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    fp16=True,
    report_to=['tensorboard'],
    logging_dir=config['logging_dir'],
    logging_steps=10,
    dataloader_num_workers=2,
    disable_tqdm=False,
    push_to_hub=False,
    seed=42,
    temperature=config['temperature'],
    label_names=['labels', 'input_text']
)

# Early stopping callback
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=2):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        val_loss = metrics.get('eval_loss', None)
        if val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print('Early stopping triggered.')
                    control.should_training_stop = True

# Trainer
trainer = DistillTrainer(
    model=student,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    teacher=teacher,
    callbacks=[EarlyStoppingCallback(patience=config['early_stopping_patience'])]
)

# Train
trainer.train()

# Save model in Ollama-compatible format
student.save_pretrained(config['export_dir'])
tokenizer.save_pretrained(config['export_dir'])

# Save training metrics
metrics = trainer.state.log_history
with open(os.path.join(config['output_dir'], 'training_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print('Distillation complete. Model exported to', config['export_dir'])
