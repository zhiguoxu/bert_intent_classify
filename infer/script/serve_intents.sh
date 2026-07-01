# 启动 intents 模型的推理服务
# 用法（在项目根目录执行）: bash infer/script/serve_intents.sh
# fuser -k -9 10001/tcp

MODEL_DIR=models/intents_onnx nohup conda run -n bert_intent_classify --no-capture-output \
  uvicorn infer:app --app-dir infer --host 0.0.0.0 --port 10001 --workers 4 \
  > infer_intents.log 2>&1 &
