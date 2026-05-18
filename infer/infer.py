import os
from typing import List
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import onnxruntime as ort
from transformers import BertTokenizerFast


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
    model_dir = "../models/bert_onnx"
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
    return {"status": "healthy"}


"""
curl -X 'POST' \
  'http://8.145.38.125:10001/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
    "这是一个非常高效的 BERT 部署方案。",
    "人工智能改变世界。"
  ]
}'
"""
