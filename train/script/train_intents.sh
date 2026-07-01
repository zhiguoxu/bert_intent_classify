#!/usr/bin/env bash
# 切到项目根目录，保证 train/train.py 相对路径正确
cd "$(dirname "$0")/../.." || exit 1

DATASET=intents

# 日志输出到该数据集对应的 output 目录
LOG="$(pwd)/output/$DATASET/train_intents.log"
mkdir -p "$(dirname "$LOG")"

nohup conda run -n bert_classify --no-capture-output \
  python train/train.py "$DATASET" > "$LOG" 2>&1 &

echo "日志输出到: $LOG"
