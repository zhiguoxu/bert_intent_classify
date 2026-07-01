# 将 intents 数据集训练出的最新模型导出为 ONNX
# 用法（在项目根目录执行）: bash infer/script/convert_intents_model.sh
set -e

DATASET=intents

# 自动选取 output/<dataset>/ 下最新的一个训练产物目录
MODEL_DIR=$(ls -dt output/${DATASET}/model_* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then
  echo "未找到 output/${DATASET}/model_* 训练产物，请先训练。"
  exit 1
fi

ONNX_DIR=models/${DATASET}_onnx
echo "导出模型: $MODEL_DIR -> $ONNX_DIR"

rm -rf "$ONNX_DIR"
conda run -n bert_intent_classify --no-capture-output \
  optimum-cli export onnx \
    --model "$MODEL_DIR" \
    --optimize O3 \
    --task text-classification \
    "$ONNX_DIR"

# 训练时已把 label_map.csv 存进模型目录，随模型一并拷到 onnx 目录，部署与训练两边隔离且不错配
cp "$MODEL_DIR/label_map.csv" "$ONNX_DIR/label_map.csv"
echo "完成: $ONNX_DIR (含 label_map.csv)"
