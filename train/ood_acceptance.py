"""界外语句(OOD)回归验收：检查模型对"21 类之外的祈使句/闲聊"是否正确兜底到 other。

背景：2026-07-14 探测发现模型把大量未定义指令(握手/抖腿/蹲下/用户自述"我去客厅了"等)
以 ≥95% 置信度错归到动作类，会穿透生产阈值触发误动作。语料按六类失败模式整理成
data/acceptance/ood_probes.csv，其中也混入定义内意图的换说法作为正例对照——
防止补 other 负例把正类召回打伤（打地鼠工艺，见记忆 key-extractor 形态迭代）。

⚠️ 验收句不得进入训练语料（data/intents/*.txt），补数据前先与本文件的 CSV 去重。

CSV 格式: text, expect, group
  expect 竖线分隔多个可接受标签(语义两可句放宽)，命中任一即通过。

判定（对齐生产逻辑 bert_classifier.py: label!=other 且 conf>=0.95 才执行）:
  PASS      预测命中 expect 之一
  FAIL-hard 预测越界且置信度 >= 0.95 —— 会真实触发误动作，重训必须清零的目标
  FAIL-soft 预测越界但置信度 < 0.95 —— 被生产阈值挡住，观察项

用法:
  python train/ood_acceptance.py [dataset] [model_dir]   # 本地模型评测(训练机/GPU 服务机)
  python train/ood_acceptance.py --url http://IP:10002   # 打在线服务评测
  model_dir 默认取 output/<dataset>/ 下最新 model_* 目录。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

PROD_THRESHOLD = 0.95
BATCH_SIZE = 64
ACCEPTANCE_CSV = Path(__file__).parent / "data" / "acceptance" / "ood_probes.csv"


class ProbeRow(TypedDict):
    text: str
    expect: List[str]
    group: str


def load_probes() -> List[ProbeRow]:
    rows: List[ProbeRow] = []
    with open(ACCEPTANCE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(ProbeRow(
                text=row["text"],
                expect=row["expect"].split("|"),
                group=row["group"],
            ))
    return rows


def classify_via_url(base_url: str, texts: List[str]) -> List[Tuple[str, float]]:
    """打在线推理服务，label_map 用 infer/label_map.csv（与 10002 模型配套）。"""
    import httpx
    import numpy as np

    label_map_path = Path(__file__).parent.parent / "infer" / "label_map.csv"
    id2label: Dict[int, str] = {}
    with open(label_map_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id2label[int(row["label"])] = row["category"]

    results: List[Tuple[str, float]] = []
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0) as client:
        for i in range(0, len(texts), BATCH_SIZE):
            resp = client.post("/predict", json={"texts": texts[i: i + BATCH_SIZE]})
            resp.raise_for_status()
            logits = np.array(resp.json()["logits"])
            shifted = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            for idx, conf in zip(probs.argmax(axis=1), probs.max(axis=1)):
                results.append((id2label[int(idx)], float(conf)))
    return results


def classify_via_model(model_dir: Path, texts: List[str]) -> List[Tuple[str, float]]:
    """本地加载训练产物推理，label_map 用模型目录自带的（与该模型训练时一致）。"""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    id2label: Dict[int, str] = {}
    with open(model_dir / "label_map.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id2label[int(row["label"])] = row["category"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    results: List[Tuple[str, float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tokenizer(
                texts[i: i + BATCH_SIZE],
                truncation=True, max_length=512, padding=True, return_tensors="pt",
            ).to(device)
            probs = F.softmax(model(**enc).logits.float().cpu(), dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            for p, c in zip(preds, confs):
                results.append((id2label[int(p)], float(c)))
    return results


def report(rows: List[ProbeRow], preds: List[Tuple[str, float]]) -> int:
    """打印验收报告，返回 FAIL-hard 数量（作为退出码，0 = 验收通过）。"""
    hard: List[Tuple[ProbeRow, str, float]] = []
    soft: List[Tuple[ProbeRow, str, float]] = []
    group_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [pass, total]

    for row, (label, conf) in zip(rows, preds):
        ok = label in row["expect"]
        group_stats[row["group"]][1] += 1
        if ok:
            group_stats[row["group"]][0] += 1
        elif conf >= PROD_THRESHOLD:
            hard.append((row, label, conf))
        else:
            soft.append((row, label, conf))

    total = len(rows)
    n_pass = total - len(hard) - len(soft)
    print("=" * 60)
    print(f"OOD 验收: {total} 条 | PASS {n_pass} | "
          f"FAIL-hard {len(hard)} (穿透生产阈值) | FAIL-soft {len(soft)}")
    print("=" * 60)

    print("\n--- 按组通过率 ---")
    for group, (p, t) in sorted(group_stats.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        mark = "  " if p == t else "⚠️"
        print(f"{mark} {group}: {p}/{t}")

    if hard:
        print(f"\n--- FAIL-hard: 会真实触发误动作，共 {len(hard)} 条 ---")
        for row, label, conf in sorted(hard, key=lambda x: -x[2]):
            print(f"🔴 [{row['group']}] {row['text']!r} → {label} ({conf:.2%}), "
                  f"预期 {'/'.join(row['expect'])}")
    if soft:
        print(f"\n--- FAIL-soft: 被 {PROD_THRESHOLD:.0%} 阈值挡住，共 {len(soft)} 条 ---")
        for row, label, conf in sorted(soft, key=lambda x: -x[2]):
            print(f"🟡 [{row['group']}] {row['text']!r} → {label} ({conf:.2%}), "
                  f"预期 {'/'.join(row['expect'])}")
    return len(hard)


def main() -> None:
    rows = load_probes()
    texts = [r["text"] for r in rows]

    if "--url" in sys.argv:
        base_url = sys.argv[sys.argv.index("--url") + 1]
        print(f"评测在线服务: {base_url} | 验收集: {ACCEPTANCE_CSV.name} ({len(rows)} 条)")
        preds = classify_via_url(base_url, texts)
    else:
        dataset = sys.argv[1] if len(sys.argv) > 1 else "intents"
        if len(sys.argv) > 2:
            model_dir = Path(sys.argv[2])
        else:
            out_root = Path(__file__).resolve().parent.parent / "output" / dataset
            candidates = sorted(out_root.glob("model_*"))
            if not candidates:
                raise FileNotFoundError(f"未找到 {out_root}/model_* 训练产物，请先训练。")
            model_dir = candidates[-1]
        print(f"评测本地模型: {model_dir} | 验收集: {ACCEPTANCE_CSV.name} ({len(rows)} 条)")
        preds = classify_via_model(model_dir, texts)

    sys.exit(1 if report(rows, preds) > 0 else 0)


if __name__ == "__main__":
    main()
