#!/usr/bin/env bash
# 训练指定数据集(训练目标), 超参/基座由 train/dataset_configs.py 按数据集决定。
# 用法:
#   bash train_intents.sh                # 默认 intents(现网 21 类意图)
#   bash train_intents.sh vision_gate    # 视觉门控二分类
#   CUDA_ID=1 bash train_intents.sh vision_gate   # 指定显卡(开训前 nvidia-smi 挑空卡)
# 切到项目根目录，保证 train/train.py 相对路径正确
cd "$(dirname "$0")/../.." || exit 1

DATASET="${1:-intents}"

# 日志输出到该数据集对应的 output 目录
LOG="$(pwd)/output/$DATASET/train_${DATASET}.log"
mkdir -p "$(dirname "$LOG")"

nohup conda run -n bert_classify --no-capture-output \
  python train/train.py "$DATASET" > "$LOG" 2>&1 &

echo "已启动训练: dataset=$DATASET"
echo "日志输出到: $LOG"
