#!/usr/bin/env bash
# 视觉门控服务(端口 10004, 部署在 person_id 感知服务机)。
# 等价于: bash serve_intents.sh vision_gate 10004
# 模型: models/vision_gate_onnx(convert_intents_model.sh vision_gate 导出,
#       或直接拷 voice_agent/test/vision_gate/results/ft_bert-base__v3 的 ONNX 产物)
exec bash "$(dirname "$0")/serve_intents.sh" vision_gate "${PORT:-10004}"
