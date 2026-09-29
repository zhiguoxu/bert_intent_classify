#!/usr/bin/env bash
# 将指定数据集训练出的最新模型导出为 ONNX（任何目录下均可执行）
# 用法:
#   bash convert_intents_model.sh                # 默认 intents -> models/intents_onnx
#   bash convert_intents_model.sh vision_gate    # -> models/vision_gate_onnx
#   CONDA_ENV=bert_train bash convert_intents_model.sh vision_gate
#       # person_id 机: optimum/torch 装在 bert_train env(bert_classify 只有 CPU 服务最小集)
# 注意: 会先删掉 models/<dataset>_onnx 再导出; 在线服务已把模型读进内存不受影响,
# 但要保留回滚点就先把旧目录改名(如 vision_gate_onnx_v4_20260826)。
# 切到项目根目录，保证下面的相对路径正确
cd "$(dirname "$0")/../.." || exit 1
set -e

DATASET="${1:-intents}"
CONDA_ENV="${CONDA_ENV:-bert_classify}"

# 自动选取 output/<dataset>/ 下最新的一个训练产物目录
MODEL_DIR=$(ls -dt output/${DATASET}/model_* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then
  echo "未找到 output/${DATASET}/model_* 训练产物，请先训练。"
  exit 1
fi

ONNX_DIR=models/${DATASET}_onnx
echo "导出模型: $MODEL_DIR -> $ONNX_DIR"

rm -rf "$ONNX_DIR"
conda run -n "$CONDA_ENV" --no-capture-output \
  optimum-cli export onnx \
    --model "$MODEL_DIR" \
    --optimize O3 \
    --task text-classification \
    "$ONNX_DIR"

# 训练时已把 label_map.csv 存进模型目录，随模型一并拷到 onnx 目录，部署与训练两边隔离且不错配
cp "$MODEL_DIR/label_map.csv" "$ONNX_DIR/label_map.csv"
echo "完成: $ONNX_DIR (含 label_map.csv)"
