# vision_gate 语料(导出产物, 勿手改)

源头: voice_agent/test/vision_gate/data/raw_{positive,negative}.py(按场景维护), 由 export_to_pipeline.py 合并导出(train+dev 切分)。
改语料请改源头后重新导出; 评测集(test/困难集/家庭专项集)刻意不在此,
流水线训完后用 voice_agent/test/vision_gate/eval_pipeline_model.py 质检。
