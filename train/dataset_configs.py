"""数据集(训练目标)注册表 —— 一处配置, prepare/train/脚本共用。

设计约定:
  - 一个"数据集"= 一个训练目标, 语料放 train/data/<dataset>/*.txt(一个 txt = 一个类,
    文件名即类名), 产物按数据集隔离在 output/<dataset>/。
  - 新增训练目标只需: ① 建语料目录; ② 在 DATASET_CONFIGS 里登记一条(选基座与超参);
    ③ prepare -> train -> convert -> serve 四步命令全部带同一个数据集名。
  - 未登记的数据集使用 DatasetConfig 默认值, 与 intents 的历史行为完全一致,
    保证既有任务零变化。

基座模型选型经验(见 voice_agent/test/vision_gate/REPORT.md):
  - 类别多/语义细(如 21 类动作意图): roberta-wwm-ext-large, 精度优先;
  - 二分类/低延迟门控(如 vision_gate): bert-wwm-ext(base), CPU ONNX fp32 单条
    p50 16~19ms; 更小的蒸馏模型(minirbt)在多样语料上有容量瓶颈, 不建议。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    # 基座模型: 相对工程根的路径或绝对路径(服务器上用 train/script/download_model.sh 下载)
    base_model: str = "models/chinese-roberta-wwm-ext-large"
    lr: float = 2e-5
    epochs: int = 12
    batch_size: int = 16
    eval_batch_size: int = 16
    max_length: int = 512
    # 每类语料的封顶条数(防大类主导); None = 全量不截断
    max_samples_per_class: int | None = 250
    # 不受封顶限制的类(如 intents 的兜底类 other 要覆盖闲聊+界外祈使句, 全量入库)
    uncapped_classes: tuple = ("other",)
    description: str = ""


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    # 21 类机器人动作意图(现网任务, 参数即历史默认值)
    "intents": DatasetConfig(
        description="21 类机器人动作意图 + other 兜底(现网 10001/10002)",
    ),
    # 视觉门控: 这一轮对话是否需要摄像头画面(vision / no_vision)
    # 语料源头在 voice_agent/test/vision_gate/data/(按场景维护),
    # 用 export_to_pipeline.py 导出为两个 txt; 评测用那边的困难集/家庭专项集(只测不训)。
    "vision_gate": DatasetConfig(
        base_model="models/chinese-bert-wwm-ext",
        lr=2e-5,
        epochs=8,
        batch_size=32,
        # 带上文的双句样本(本轮+上轮问答, 指代消解方向)约 130 字符,
        # 与 agent_server/agent/vision_gate/context_format.py 的
        # GATE_MAX_LENGTH 对齐; 单轮短句照常, padding 不影响精度
        max_length=160,
        max_samples_per_class=None,   # 二分类语料已人工配平, 不截断
        uncapped_classes=(),
        description="视觉门控二分类(部署: person_id 服务机 10004 端口)",
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """取数据集配置; 未登记的返回默认值(与 intents 历史行为一致)。"""
    cfg = DATASET_CONFIGS.get(name)
    if cfg is None:
        print(f"[dataset_configs] '{name}' 未登记, 使用默认配置(同 intents 历史行为)")
        cfg = DatasetConfig()
    return cfg
