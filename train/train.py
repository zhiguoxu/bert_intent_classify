import os
from pathlib import Path

cuda_id = 7
# 只使用第一个 GPU (索引为 0)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_id}"

from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

from datasets import load_dataset

seed = 42
model_name_or_path = Path(__file__).parent.parent / "models/chinese-roberta-wwm-ext-large"
output_dir = Path(__file__).parent.parent / f"output/intent_classify_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
max_length = 512
num_labels = 20
lr = 2e-5
batch_size = 16
eval_batch_size = 16
epochs = 12
fp16 = True
device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

set_seed(seed)

train_file = str(Path(__file__).parent.parent / "output/train_data.csv")
valid_file = str(Path(__file__).parent.parent / "output/train_data.csv")
raw_datasets = load_dataset("csv", data_files={"train": train_file, "validation": valid_file})

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def tokenize_function(examples):
    # examples["text"] can be a list
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=True, return_tensors="pt")


tokenized = raw_datasets.map(
    tokenize_function,
    batched=True,
    batch_size=8,
    remove_columns=[c for c in raw_datasets["train"].column_names if c not in ("text", "label")],
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, num_labels=num_labels
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=False,
    bf16=False,
    save_total_limit=3,
    push_to_hub=False,
)


def compute_metrics(pred):
    labels = pred.label_ids
    predictions = pred.predictions.argmax(-1)
    f1 = f1_score(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model()
