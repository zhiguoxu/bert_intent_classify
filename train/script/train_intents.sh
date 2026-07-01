#!/usr/bin/env bash
# 日志写到调用时所在目录的 logs/
LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_intents.log"

# 切到项目根目录再执行，保证 train/train.py 相对路径正确
cd "$(dirname "$0")/../.." || exit 1

nohup /home/zhiguo/miniconda3/bin/conda run -n bert_intent_classify --no-capture-output \
  python train/train.py intents > "$LOG" 2>&1 &

echo "日志输出到: $LOG"
