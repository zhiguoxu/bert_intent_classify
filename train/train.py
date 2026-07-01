import os
import sys
import csv
import shutil
from pathlib import Path

cuda_id = 3
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

# ============ 数据集 / 任务 ============
# 不同分类任务使用不同 dataset 名称，训练数据与模型产物按名称隔离：
#   原始语料:   train/data/<dataset>/*.txt
#   预处理产物: output/<dataset>/{train_data.csv, label_map.csv}
#   模型产物:   output/<dataset>/model_<时间戳>/
# 用法: python train/train.py [dataset]   (默认 "intents"，对应 train/data/intents)
dataset = sys.argv[1] if len(sys.argv) > 1 else "intents"

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "output" / dataset

model_name_or_path = PROJECT_ROOT / "models/chinese-roberta-wwm-ext-large"
output_dir = DATASET_DIR / f"model_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
train_file = str(DATASET_DIR / "train_data.csv")
valid_file = str(DATASET_DIR / "train_data.csv")
label_map_file = DATASET_DIR / "label_map.csv"

max_length = 512
lr = 2e-5
batch_size = 16
eval_batch_size = 16
epochs = 12
fp16 = True
device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

set_seed(seed)


def load_label_map(path):
    """从 label_map.csv 读取 {label_id: category} 映射"""
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[int(row["label"])] = row["category"]
    return m


# 类别数量与 id<->name 映射均由该数据集的 label_map 决定，
# 避免硬编码 num_labels 与不同任务错配
if not label_map_file.exists():
    raise FileNotFoundError(
        f"未找到 {label_map_file}，请先运行: python train/prepare_train_data.py {dataset}"
    )
id2label = load_label_map(label_map_file)
label2id = {name: idx for idx, name in id2label.items()}
num_labels = len(id2label)
print(f"[dataset={dataset}] num_labels={num_labels} | 产物目录: {DATASET_DIR}")

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
    model_name_or_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
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
    # 把该数据集的 label_map 一并存入模型目录：模型自带标签映射，
    # 部署时随模型一起拷贝到 infer/，两边隔离且永不错配
    shutil.copy(label_map_file, output_dir / "label_map.csv")

    report_lines = []
    def log_and_print(msg):
        print(msg)
        report_lines.append(msg)

    log_and_print("\n" + "="*50)
    log_and_print("Evaluating validation set and analyzing worst cases...")
    log_and_print("="*50)

    import torch.nn.functional as F
    import numpy as np

    # 复用模块加载好的映射表（已由 dataset 的 label_map.csv 得到）
    def get_label_name(lbl_id):
        return id2label.get(int(lbl_id), str(lbl_id))

    # 在验证集上进行预测
    pred_output = trainer.predict(tokenized["validation"])
    logits = torch.tensor(pred_output.predictions)
    labels = torch.tensor(pred_output.label_ids)

    # 计算每个样本的 cross entropy loss
    losses = F.cross_entropy(logits, labels, reduction='none')

    # 计算预测概率
    probs = F.softmax(logits, dim=-1)
    max_probs, preds = torch.max(probs, dim=-1)

    losses = losses.numpy()
    preds = preds.numpy()
    max_probs = max_probs.numpy()
    labels = labels.numpy()

    # 按照 loss 降序排列
    sorted_indices = np.argsort(-losses)

    top_10_indices = sorted_indices[:10]
    top_10_set = set(top_10_indices)

    val_raw_data = raw_datasets["validation"]

    log_and_print("\n--- Loss 最大的前 10 个数据 ---")
    for i, idx in enumerate(top_10_indices):
        if idx >= len(val_raw_data): continue
        text = val_raw_data[int(idx)]["text"]
        true_lbl = labels[idx]
        pred_lbl = preds[idx]
        prob = max_probs[idx]
        loss = losses[idx]

        true_name = get_label_name(true_lbl)
        pred_name = get_label_name(pred_lbl)
        match_str = "✅" if true_lbl == pred_lbl else "❌"

        log_and_print(f"[{i+1}] Loss: {loss:.4f} | 真实 Label: {true_name} | 预测 Label: {pred_name} | 预测概率: {prob:.4f} {match_str}")
        log_and_print(f"文本: {text}\n")

    log_and_print("\n--- 其他预测不一致的数据 (不在 Loss 前 10 名中) ---")
    misclassified_indices = np.where(preds != labels)[0]
    other_misclassified = [idx for idx in misclassified_indices if idx not in top_10_set]

    if len(other_misclassified) == 0:
        log_and_print("无。")
    else:
        for idx in other_misclassified:
            if idx >= len(val_raw_data): continue
            text = val_raw_data[int(idx)]["text"]
            true_lbl = labels[idx]
            pred_lbl = preds[idx]
            prob = max_probs[idx]
            loss = losses[idx]

            true_name = get_label_name(true_lbl)
            pred_name = get_label_name(pred_lbl)

            log_and_print(f"Loss: {loss:.4f} | 真实 Label: {true_name} | 预测 Label: {pred_name} | 预测概率: {prob:.4f} ❌")
            log_and_print(f"文本: {text}\n")

    report_file = output_dir / "valid_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n验证报告已保存至: {report_file}")

