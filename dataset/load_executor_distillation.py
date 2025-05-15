from datasets import load_dataset

train_ds = load_dataset('json', data_files='dataset/executor_distillation_train.json')['train']
val_ds = load_dataset('json', data_files='dataset/executor_distillation_val.json')['train']

print(f"Train set size: {len(train_ds)}")
print(f"Validation set size: {len(val_ds)}")
print("Sample train example:", train_ds[0])
print("Sample val example:", val_ds[0])
