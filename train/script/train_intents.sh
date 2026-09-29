#!/usr/bin/env bash
# 训练指定数据集(训练目标), 超参/基座由 train/dataset_configs.py 按数据集决定。
# 用法:
#   bash train_intents.sh                # 默认 intents(现网 21 类意图)
#   bash train_intents.sh vision_gate    # 视觉门控二分类
#   CUDA_ID=1 bash train_intents.sh vision_gate   # 指定显卡(开训前 nvidia-smi 挑空卡)
#   CONDA_ENV=bert_train bash train_intents.sh vision_gate
#       # person_id 机的 bert_classify 只装了 CPU 服务最小集, 训练用同机 bert_train env
# 切到项目根目录，保证 train/train.py 相对路径正确
cd "$(dirname "$0")/../.." || exit 1

DATASET="${1:-intents}"
CONDA_ENV="${CONDA_ENV:-bert_classify}"

# 日志输出到该数据集对应的 output 目录
LOG="$(pwd)/output/$DATASET/train_${DATASET}.log"
mkdir -p "$(dirname "$LOG")"

nohup conda run -n "$CONDA_ENV" --no-capture-output \
  python train/train.py "$DATASET" > "$LOG" 2>&1 &

echo "已启动训练: dataset=$DATASET env=$CONDA_ENV"
echo "日志输出到: $LOG"
