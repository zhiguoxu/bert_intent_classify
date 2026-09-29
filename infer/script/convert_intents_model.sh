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
# 模型身份卡 model_info.json: 训练侧 train_info.json(数据集/基座/语料量/训练时刻) + 来源目录与
# 导出时刻; 服务 /health 原样上报, 控制台「系统配置」顶部据此显示模型版本(= 来源目录的时间戳)
# (conda run 不透传 stdin, 代码经 -c 传入)
conda run -n "$CONDA_ENV" --no-capture-output python -c "$(cat <<'EOF'
import json, sys, os, datetime
src, dst = sys.argv[1], sys.argv[2]
info = {}
p = os.path.join(src, "train_info.json")
if os.path.exists(p):
    info = json.load(open(p, encoding="utf-8"))
info.update(source_model=os.path.basename(src.rstrip("/")),
            exported_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
json.dump(info, open(os.path.join(dst, "model_info.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
EOF
)" "$MODEL_DIR" "$ONNX_DIR"
echo "完成: $ONNX_DIR (含 label_map.csv / model_info.json)"
