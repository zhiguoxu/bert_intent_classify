#!/usr/bin/env bash
# 启动 intents 模型的推理服务（任何目录下均可执行）
# 切到项目根目录，保证下面的相对路径正确
cd "$(dirname "$0")/../.." || exit 1
# fuser -k -9 10001/tcp

DATASET=intents

# 日志输出到该数据集对应的 output 目录
LOG="$(pwd)/output/$DATASET/serve_intents.log"
mkdir -p "$(dirname "$LOG")"

MODEL_DIR=models/${DATASET}_onnx nohup conda run -n bert_classify --no-capture-output \
  uvicorn infer:app --app-dir infer --host 0.0.0.0 --port 10001 --workers 4 \
  > "$LOG" 2>&1 &

echo "日志输出到: $LOG"
