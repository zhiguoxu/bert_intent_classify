# bert_intent_classify —— 多训练目标的文本分类流水线

一套"语料 txt → 训练 → ONNX 导出 → CPU/GPU 推理服务"的流水线, 支持多个独立的
训练目标(数据集), 语料、超参、产物、端口全部按数据集隔离。

## 训练目标一览

| 数据集 | 任务 | 基座 | 服务端口 | 部署机 |
|---|---|---|---|---|
| intents | 21 类机器人动作意图 + other 兜底 | roberta-wwm-ext-large | 10001(外部快照) / 10002(agent_server) | 训练机 / GPU 服务机 |
| vision_gate | 视觉门控: 本轮是否需要摄像头画面 | bert-wwm-ext(base) | **10004** | **person_id 感知服务机** |

新增训练目标三步: ① 语料放 `train/data/<dataset>/`(一个 txt = 一个类, 文件名即类名);
② 在 `train/dataset_configs.py` 登记基座与超参(不登记则用 intents 的历史默认值);
③ 下面四条命令全部带数据集名。

## 全流程命令

```bash
# 0. 基座模型(首次): 下载后放到工程根 models/ 下
bash train/script/download_model.sh hfl/chinese-bert-wwm-ext   # vision_gate 基座
# 1. 预处理: train/data/<dataset>/*.txt -> output/<dataset>/train_data.csv
python train/prepare_train_data.py vision_gate
# 2. 训练(GPU 机上用脚本; CUDA_ID 选卡): 产物 output/<dataset>/model_<时间戳>/
CUDA_ID=1 bash train/script/train_intents.sh vision_gate
# 3. 导出 ONNX: output/<dataset>/model_* 最新 -> models/<dataset>_onnx
bash infer/script/convert_intents_model.sh vision_gate
# 4. 起服务(端口约定见 serve_intents.sh 头部注释)
bash infer/script/serve_vision_gate.sh        # = serve_intents.sh vision_gate 10004
```

## vision_gate 部署(person_id 感知服务机, 端口 10004)

vision_gate 的语料源头与选型实验在 `voice_agent/test/vision_gate/`(REPORT.md 有
完整评测: bert-base fp32 家庭人群专项集 98.1%、漏视觉 0%、CPU 单条 p50 16~19ms)。

首次部署步骤(在 person_id 服务机上):

```bash
# 1. 同步本仓库到服务机(models/ 与 output/ 不入 git, 需单独同步或现场生成)
# 2. conda 环境: bert_classify(train/infer 通用), 依赖见 infer/infer_requirements.txt
#    (GPU 机装 onnxruntime-gpu, infer.py 会自动优先用 CUDAExecutionProvider)
# 3. 模型就位, 二选一:
#    a) 现场重训: 上面的全流程命令 0~3
#    b) 直接拷实验产物: voice_agent/test/vision_gate/results/ft_bert-base__v3/
#       整目录(model.onnx + tokenizer + label_map 需补拷)到 models/vision_gate_onnx/
# 4. 起服务并验证
bash infer/script/serve_vision_gate.sh
curl -X POST http://127.0.0.1:10004/predict -H 'Content-Type: application/json' \
  -d '{"texts": ["看看这是什么", "明天天气怎么样"]}'
# 返回 logits, argmax: 0=no_vision 1=vision(见 models/vision_gate_onnx/label_map.csv)
```

训完质检(只测不训的三套评测集, 在 voice_agent 仓库):

```bash
python voice_agent/test/vision_gate/eval_pipeline_model.py            # 最新训练产物
python voice_agent/test/vision_gate/eval_pipeline_model.py models/vision_gate_onnx  # 服务用 ONNX
```

改语料请改 `voice_agent/test/vision_gate/data/raw_*.py`(按场景维护)后执行
`export_to_pipeline.py` 重新导出——`train/data/vision_gate/` 是导出产物, 勿手改。

## 备注

远程解释器排除目录：
/Users/xuzhiguo/workspace/python/lx/bert_intent_classify/models
/Users/xuzhiguo/workspace/python/lx/bert_intent_classify/.git
勾选: "本地删除后删除远程"
