#!/usr/bin/env bash
# 自动切到项目根目录，无论从哪里执行都能跑
cd "$(dirname "$0")/../.." || exit 1

nohup /home/zhiguo/miniconda3/bin/conda run -n bert_intent_classify --no-capture-output \
  python train/train.py intents > "train_intents.log" 2>&1 &
