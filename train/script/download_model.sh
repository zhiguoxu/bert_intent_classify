#!/usr/bin/env bash
# 服务器无法直连 huggingface.co（Network is unreachable），使用国内镜像 hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com

hf download hfl/chinese-roberta-wwm-ext-large --local-dir ./chinese-roberta-wwm-ext-large --exclude "*.h5" --exclude "*.msgpack"
