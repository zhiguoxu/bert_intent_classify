"""对训练产物做验证集评测并生成报告。

记录内容（每次训练后自动生成，也可对任意历史模型目录单独执行）：
  - 整体指标: eval_loss / accuracy / f1（写入报告头部，便于跨版本对比）
  - loss 最大的前 TOP_K 条样本及其预测概率（观察高 loss 样本置信度是否随版本下降）
  - eval_details.csv: 逐样本明细 (text, true_label, pred_label, prob, loss)，
    跨版本对比时可按 text 关联，看同一条样本的概率变化

用法: python train/eval_report.py [dataset] [model_dir]
  dataset  默认 "intents"
  model_dir 默认取 output/<dataset>/ 下最新的 model_* 目录
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TOP_K = 20


def load_label_map(path):
    """从 label_map.csv 读取 {label_id: category} 映射"""
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[int(row["label"])] = row["category"]
    return m


def evaluate_and_report(model_dir, dataset="intents", batch_size=64, max_length=512):
    model_dir = Path(model_dir)
    project_root = Path(__file__).resolve().parent.parent
    data_file = project_root / "output" / dataset / "train_data.csv"
    # 模型目录自带 label_map.csv，保证评测标签与该模型训练时一致
    id2label = load_label_map(model_dir / "label_map.csv")

    texts, labels = [], []
    with open(data_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            all_logits.append(model(**enc).logits.float().cpu())
    logits = torch.cat(all_logits)
    labels_arr = np.array(labels)

    losses = F.cross_entropy(logits, torch.tensor(labels), reduction="none").numpy()
    probs = F.softmax(logits, dim=-1)
    max_probs, preds = torch.max(probs, dim=-1)
    preds, max_probs = preds.numpy(), max_probs.numpy()

    eval_loss = float(losses.mean())
    acc = accuracy_score(labels_arr, preds)
    f1 = f1_score(labels_arr, preds, average="weighted")

    details_file = model_dir / "eval_details.csv"
    with open(details_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "true_label", "pred_label", "prob", "loss"])
        for t, tl, pl, p, l in zip(texts, labels, preds, max_probs, losses):
            w.writerow([t, id2label[int(tl)], id2label[int(pl)], f"{p:.6f}", f"{l:.6f}"])

    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    top = np.argsort(-losses)[:TOP_K]

    log("=" * 50)
    log(f"模型: {model_dir.name} | 数据集: {dataset} | 样本数: {len(texts)}")
    log(f"整体指标: eval_loss={eval_loss:.6f} | accuracy={acc:.6f} | f1={f1:.6f}")
    log(
        f"Top{TOP_K} 高 loss 样本: 平均 loss={float(losses[top].mean()):.6f}"
        f" | 平均预测概率={float(max_probs[top].mean()):.4f}"
        f" | 误分类总数={int((preds != labels_arr).sum())}"
    )
    log("=" * 50)

    # 按类别拆分指标：类别样本数不均衡时，观察小类是否被大类挤压
    log("\n--- 按类别指标 (precision / recall / f1 / support) ---")
    log(
        classification_report(
            labels_arr,
            preds,
            labels=sorted(id2label),
            target_names=[id2label[i] for i in sorted(id2label)],
            digits=4,
            zero_division=0,
        )
    )

    log(f"\n--- Loss 最大的前 {TOP_K} 个数据 ---")
    for i, idx in enumerate(top):
        ok = "✅" if preds[idx] == labels_arr[idx] else "❌"
        log(
            f"[{i + 1}] Loss: {losses[idx]:.4f} | 真实 Label: {id2label[int(labels_arr[idx])]}"
            f" | 预测 Label: {id2label[int(preds[idx])]} | 预测概率: {max_probs[idx]:.4f} {ok}"
        )
        log(f"文本: {texts[idx]}\n")

    top_set = set(int(i) for i in top)
    others = [int(i) for i in np.where(preds != labels_arr)[0] if int(i) not in top_set]
    log(f"\n--- 其他预测不一致的数据 (不在 Loss 前 {TOP_K} 名中) ---")
    if not others:
        log("无。")
    for idx in others:
        log(
            f"Loss: {losses[idx]:.4f} | 真实 Label: {id2label[int(labels_arr[idx])]}"
            f" | 预测 Label: {id2label[int(preds[idx])]} | 预测概率: {max_probs[idx]:.4f} ❌"
        )
        log(f"文本: {texts[idx]}\n")

    report_file = model_dir / "valid_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n验证报告已保存至: {report_file}")
    print(f"逐样本明细已保存至: {details_file}")
    return {"eval_loss": eval_loss, "accuracy": acc, "f1": f1}


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "intents"
    if len(sys.argv) > 2:
        mdir = Path(sys.argv[2])
    else:
        out_root = Path(__file__).resolve().parent.parent / "output" / ds
        candidates = sorted(out_root.glob("model_*"))
        if not candidates:
            raise FileNotFoundError(f"未找到 {out_root}/model_* 训练产物，请先训练。")
        mdir = candidates[-1]
    evaluate_and_report(mdir, ds)
