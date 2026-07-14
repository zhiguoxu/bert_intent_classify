import os
import sys
import csv
import shutil
from pathlib import Path

# 用哪张卡由 CUDA_ID 环境变量指定(训练机各卡占用波动大, 开训前先 nvidia-smi 挑空卡),
# 例: CUDA_ID=1 bash train/script/train_intents.sh
cuda_id = int(os.environ.get("CUDA_ID", "3"))
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
# CUDA_VISIBLE_DEVICES 已把选中的卡映射为唯一可见设备, torch 侧永远是 cuda:0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 类别样本数不均衡（如 navigate 210 条 vs 多数类 ~50 条）时，
# 按频率倒数加权交叉熵: weight_c = N / (K * n_c)，避免多数类主导梯度
from collections import Counter
import torch.nn.functional as F

label_counts = Counter(raw_datasets["train"]["label"])
total = sum(label_counts.values())
class_weights = torch.tensor(
    [total / (num_labels * label_counts[i]) for i in range(num_labels)],
    dtype=torch.float,
)
print(f"类别权重: { {id2label[i]: round(float(w), 3) for i, w in enumerate(class_weights)} }")


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits, labels, weight=class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss

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
    metric_for_best_model="loss",
    greater_is_better=False,
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


trainer = WeightedTrainer(
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

    # 评测报告统一由 eval_report 生成：整体 eval_loss/accuracy/f1 +
    # loss 最大的前 TOP_K 条样本概率 + 逐样本明细 eval_details.csv，便于跨版本对比
    from eval_report import evaluate_and_report
    evaluate_and_report(output_dir, dataset)

