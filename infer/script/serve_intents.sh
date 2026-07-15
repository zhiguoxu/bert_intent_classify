#!/usr/bin/env bash
# 启动 intents 模型的推理服务（任何目录下均可执行）
#
# 端口约定（2026-07-15 起，训练机 / GPU 服务机通用），两端口严格隔离：
#   10001 —— 专供外部调用方，钉死带日期的模型快照目录（现为 models/intents_onnx_20260714，21 类），
#            后续重训/更新 10002 不影响它；要给 10001 换模型，先 cp 新快照再改启动参数
#   10002 —— 供 agent_server，跑 models/intents_onnx（每次重训覆盖这里），新模型一律先上这里验证
#
# 用法：
#   bash serve_intents.sh                # 默认：端口 10002 + models/intents_onnx（最新模型）
#   PORT=10001 MODEL_DIR=models/intents_onnx_20260714 bash serve_intents.sh
#                                        # 外部调用方那份（钉死快照）
#
# 切到项目根目录，保证下面的相对路径正确
cd "$(dirname "$0")/../.." || exit 1

DATASET=intents
PORT="${PORT:-10002}"
MODEL_DIR="${MODEL_DIR:-models/${DATASET}_onnx}"

if [ ! -f "$MODEL_DIR/model.onnx" ]; then
  echo "模型目录 $MODEL_DIR 下没有 model.onnx，请先用 convert_intents_model.sh 导出。"
  exit 1
fi

# 端口被占用就报错退出，避免 nohup 静默失败（旧服务需要自己 kill 掉再启动）
if ss -tln | grep -q ":${PORT} "; then
  echo "端口 ${PORT} 已被占用，请先停掉旧服务再启动。占用进程："
  ss -tlnp | grep ":${PORT} "
  exit 1
fi

# 日志按端口分开，互不覆盖
LOG="$(pwd)/output/$DATASET/serve_intents_${PORT}.log"
mkdir -p "$(dirname "$LOG")"

# --timeout-keep-alive 300: 意图分类在对话首字延迟的关键路径上, 闲置连接保得久一点,
# 配合 agent_server 侧客户端的常驻连接池(keepalive 不过期), 避免每轮对话重付 TCP 握手
# (uvicorn 默认 5s, 而对话轮距几乎总超 5s)。取舍详见 voice_agent/docs/09-latency-keepalive.md。
MODEL_DIR="$MODEL_DIR" nohup conda run -n bert_classify --no-capture-output \
  uvicorn infer:app --app-dir infer --host 0.0.0.0 --port "$PORT" --workers 4 \
  --timeout-keep-alive 300 \
  > "$LOG" 2>&1 &

echo "已启动: 端口=$PORT 模型=$MODEL_DIR"
echo "日志输出到: $LOG"
