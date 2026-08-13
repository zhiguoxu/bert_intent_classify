#!/usr/bin/env bash
# 下载基座模型到 models/ 目录（服务器无法直连 huggingface.co，使用国内镜像 hf-mirror.com）
# 用法:
#   bash download_model.sh                              # 默认 roberta-large(intents 基座)
#   bash download_model.sh hfl/chinese-bert-wwm-ext     # vision_gate 基座
export HF_ENDPOINT=https://hf-mirror.com

REPO="${1:-hfl/chinese-roberta-wwm-ext-large}"
DIR="$(basename "$REPO")"

hf download "$REPO" --local-dir "./${DIR}" --exclude "*.h5" --exclude "*.msgpack"
echo "完成: ./${DIR} (请置于工程根的 models/ 下, 与 dataset_configs.py 的 base_model 对应)"
