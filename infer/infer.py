import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import onnxruntime as ort
from transformers import BertTokenizerFast

# 时间字段统一给 naive 北京时间字面量(与 voice_agent 各服务 started_at 口径一致)
_CST = timezone(timedelta(hours=8))


def _cst_str(ts: float | None = None) -> str:
    dt = datetime.fromtimestamp(ts, _CST) if ts is not None else datetime.now(_CST)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _model_meta(model_dir: str, model_path: str) -> dict:
    """/health 里报告的模型身份: 目录、版本(model_info.json 由 convert 脚本随导出写入,
    没有则退化为 model.onnx 的修改时间)、标签表。"""
    info = {}
    info_path = os.path.join(model_dir, "model_info.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    labels = {}
    lm_path = os.path.join(model_dir, "label_map.csv")
    if os.path.exists(lm_path):
        with open(lm_path, encoding="utf-8") as f:
            labels = {row["label"]: row["category"] for row in csv.DictReader(f)}
    mtime = _cst_str(os.path.getmtime(model_path))
    # source_model 是训练产物目录名 model_<时间戳>, 去掉前缀就是版本号
    version = str(info.get("source_model", "")).removeprefix("model_") or f"mtime {mtime}"
    return {
        "model_dir": os.path.relpath(model_dir, Path(__file__).resolve().parent.parent),
        "model_version": version,
        "model_mtime": mtime,
        "model_info": info,
        "labels": labels,
    }


# 定义请求与响应格式 (Pydantic V2)
class InferenceRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=64, description="待推理的文本列表")


class InferenceResponse(BaseModel):
    logits: List[List[float]] = Field(..., description="模型输出的 Logits")


# 全局上下文管理
model_resource = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    管理服务生命周期，确保模型在服务启动时仅加载一次
    """
    # 模型目录：可用环境变量 MODEL_DIR 指定不同数据集的模型（如 models/intents_onnx）。
    # 相对路径以项目根目录为基准，与启动时的 cwd 无关。
    project_root = Path(__file__).resolve().parent.parent
    model_dir = str(project_root / os.environ.get("MODEL_DIR", "models/bert_onnx"))
    model_path = os.path.join(model_dir, "model.onnx")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未在 {model_path} 找到 ONNX 模型文件，请先执行导出命令。")

    # 1. 初始化高速 Rust Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)

    # 2. 配置 ONNX Runtime Session Options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 生产避坑：如果使用多进程(Uvicorn workers > 1)，必须限制单个 Session 的线程数，防止 CPU 爆满
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1

    # 3. 指定执行提供者 (Execution Providers)
    # 如果有 GPU 环境，优先使用 CUDAExecutionProvider
    available_providers = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider"] + providers

    session = ort.InferenceSession(model_path, sess_options, providers=providers)

    # 暂存到全局资源字典
    model_resource["tokenizer"] = tokenizer
    model_resource["session"] = session
    # /health 的身份信息一次算好: 多 worker 时各进程启动时刻略有差异, 取本进程的即可
    model_resource["meta"] = {
        **_model_meta(model_dir, model_path),
        "providers": session.get_providers(),
        "started_at": _cst_str(),
        "port": int(os.environ["PORT"]) if os.environ.get("PORT", "").isdigit() else None,
    }

    yield
    # 服务关闭时清理资源
    model_resource.clear()


app = FastAPI(title="BERT ONNX Inference Server", lifespan=lifespan)


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    try:
        tokenizer = model_resource["tokenizer"]
        session = model_resource["session"]

        # 1. 动态 Padding 编码 (仅对当前 Batch 内最长文本对齐，压榨 CPU/GPU 计算资源)
        encoded_inputs = tokenizer(
            request.texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"  # 直接返回 NumPy 数组以供 ONNX Runtime 使用
        )

        # 2. 构建 ONNX 要求的输入字典
        # 针对标准 BERT，输入包含 input_ids, attention_mask, token_type_ids
        onnx_inputs = {
            "input_ids": encoded_inputs["input_ids"].astype(np.int64),
            "attention_mask": encoded_inputs["attention_mask"].astype(np.int64)
        }
        if "token_type_ids" in encoded_inputs:
            onnx_inputs["token_type_ids"] = encoded_inputs["token_type_ids"].astype(np.int64)

        # 3. 执行推理 (ONNX Runtime 在 C++ 底层会释放 Python GIL)
        # 默认获取第一个输出节点的 Tensor (logits)
        onnx_outputs = session.run(None, onnx_inputs)
        logits = onnx_outputs[0].tolist()

        return InferenceResponse(logits=logits)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


@app.get("/health")
async def health_check():
    """保活探针 + 服务身份: status 之外附带模型目录/版本/标签表/启动时刻/端口,
    供 agent_server 转给 web 控制台「系统配置」页展示(调用方只认 status 字段也不受影响)。"""
    return {"status": "healthy", **model_resource.get("meta", {})}


"""
端口约定: 10002 = 供 agent_server(最新模型, 重训覆盖); 10001 = 专供外部调用方(钉死快照, 与 10002 隔离)。
curl -X 'POST' \
  'http://127.0.0.1:10002/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
    "这是一个非常高效的 BERT 部署方案。",
    "人工智能改变世界。"
  ]
}'
"""
