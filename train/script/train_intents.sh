#!/usr/bin/env bash
# 切到项目根目录，保证 train/train.py 相对路径正确
cd "$(dirname "$0")/../.." || exit 1

# 日志输出到项目 output 目录
LOG="$(pwd)/output/train_intents.log"
mkdir -p "$(dirname "$LOG")"

nohup /home/zhiguo/miniconda3/bin/conda run -n bert_intent_classify --no-capture-output \
  python train/train.py intents > "$LOG" 2>&1 &

echo "日志输出到: $LOG"
