"""
BERT Intent Classify - Async Client

基于 httpx 的异步客户端封装，提供：
  - 连接池复用 & Keep-Alive
  - 可配置超时 & 自动重试
  - 大批量自动拆分 (auto batching)
  - logits → label 便捷转换
  - async context manager 生命周期管理

Usage:
    async with BertIntentClassifyClient("http://localhost:8000", "label_map.csv") as client:
        # 健康检查
        ok = await client.health()

        # 原始 logits
        logits = await client.predict(["你好", "再见"])

        # 直接拿 argmax label + confidence
        results = await client.classify(["你好", "再见"])
"""

from __future__ import annotations

import asyncio
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 分类结果
# ──────────────────────────────────────────────
@dataclass
class ClassifyResult:
    """单条文本的分类结果"""
    label: str
    confidence: float

    def __repr__(self) -> str:
        return f"{self.label}({self.confidence:.2%})"


# ──────────────────────────────────────────────
# 异常定义
# ──────────────────────────────────────────────
class BertClientError(Exception):
    """客户端基础异常"""


class BertClientServerError(BertClientError):
    """服务端返回非 2xx 状态码"""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[HTTP {status_code}] {detail}")


class BertClientTimeoutError(BertClientError):
    """请求超时"""


class BertClientConnectionError(BertClientError):
    """连接失败（服务不可达）"""


# ──────────────────────────────────────────────
# 客户端配置
# ──────────────────────────────────────────────
@dataclass
class ClientConfig:
    """客户端可调参数，一次配置全局生效"""

    # 超时设置 (秒)
    connect_timeout: float = 5.0
    read_timeout: float = 30.0

    # 重试策略
    max_retries: int = 3
    retry_backoff_factor: float = 0.3  # 退避基数: 0.3s, 0.6s, 1.2s ...

    # 连接池
    max_connections: int = 50
    max_keepalive_connections: int = 10

    # 批量推理
    batch_size: int = 64  # 与服务端 max_items 对齐

    # 默认 headers
    headers: dict = field(default_factory=lambda: {"Content-Type": "application/json"})


# ──────────────────────────────────────────────
# 异步推理客户端
# ──────────────────────────────────────────────
class BertIntentClassifyClient:
    """
    BERT Intent Classify 异步客户端

    Parameters
    ----------
    base_url : str
        推理服务根地址，例如 "http://localhost:8000"
    label_map_path : str | Path
        label_map.csv 路径（由 prepare_train_data.py 生成），
        CSV 格式: label(int),category(str)
    config : ClientConfig, optional
        客户端配置，默认使用 ClientConfig 默认值
    """

    def __init__(
            self,
            base_url: str,
            label_map_path: str | Path,
            config: Optional[ClientConfig] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._config = config or ClientConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._label_map: Dict[int, str] = self._load_label_map(label_map_path)
        logger.info("已加载 %d 个类别映射: %s", len(self._label_map), self._label_map)

    # ── 标签映射 ──────────────────────────────

    @staticmethod
    def _load_label_map(path: str | Path) -> Dict[int, str]:
        """
        从 label_map.csv 加载 {int_id: category_name} 映射

        CSV 格式 (带表头):
            label,category
            0,greeting
            1,goodbye
            ...
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"label_map 文件不存在: {path}")

        label_map: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_map[int(row["label"])] = row["category"]
        return label_map

    @property
    def label_map(self) -> Dict[int, str]:
        """只读访问类别映射表"""
        return self._label_map

    # ── 生命周期 ──────────────────────────────

    def _build_client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(
            connect=self._config.connect_timeout,
            read=self._config.read_timeout,
            write=self._config.read_timeout,
            pool=self._config.connect_timeout,
        )
        limits = httpx.Limits(
            max_connections=self._config.max_connections,
            max_keepalive_connections=self._config.max_keepalive_connections,
        )
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            limits=limits,
            headers=self._config.headers,
        )

    async def connect(self) -> "BertIntentClassifyClient":
        """显式初始化底层连接池（不使用 async with 时调用此方法）"""
        self._client = self._build_client()
        return self

    async def __aenter__(self) -> "BertIntentClassifyClient":
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """显式关闭底层连接池"""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise BertClientError(
                "客户端未初始化。请使用 'async with BertIntentClassifyClient(...) as client:' "
                "或手动调用 'await client.connect()'"
            )
        return self._client

    # ── 内部工具 ──────────────────────────────

    async def _request_with_retry(
            self,
            method: str,
            path: str,
            **kwargs,
    ) -> httpx.Response:
        """
        带指数退避重试的 HTTP 请求

        仅对 连接错误 / 超时 / 5xx 进行重试，4xx 直接抛出。
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                resp = await self.client.request(method, path, **kwargs)

                # 4xx 不重试，直接报错
                if 400 <= resp.status_code < 500:
                    detail = resp.text
                    try:
                        detail = resp.json().get("detail", detail)
                    except Exception:
                        pass
                    raise BertClientServerError(resp.status_code, detail)

                # 5xx 重试
                if resp.status_code >= 500:
                    detail = resp.text
                    try:
                        detail = resp.json().get("detail", detail)
                    except Exception:
                        pass
                    last_exc = BertClientServerError(resp.status_code, detail)
                    logger.warning(
                        "服务端错误 [%d], 第 %d/%d 次重试...",
                        resp.status_code,
                        attempt,
                        self._config.max_retries,
                    )
                else:
                    return resp

            except httpx.TimeoutException as e:
                last_exc = BertClientTimeoutError(str(e))
                logger.warning("请求超时, 第 %d/%d 次重试...", attempt, self._config.max_retries)

            except httpx.ConnectError as e:
                last_exc = BertClientConnectionError(str(e))
                logger.warning("连接失败, 第 %d/%d 次重试...", attempt, self._config.max_retries)

            # 指数退避
            if attempt < self._config.max_retries:
                backoff = self._config.retry_backoff_factor * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

        # 所有重试都用完
        raise last_exc  # type: ignore[misc]

    # ── 公开 API ──────────────────────────────

    async def health(self) -> bool:
        """
        健康检查

        Returns
        -------
        bool
            服务是否正常运行
        """
        try:
            resp = await self._request_with_retry("GET", "/health")
            data = resp.json()
            return data.get("status") == "healthy"
        except BertClientError:
            return False

    async def predict(self, texts: List[str]) -> List[List[float]]:
        """
        批量推理，返回原始 logits

        自动按 batch_size 拆分大批量请求并发送。

        Parameters
        ----------
        texts : List[str]
            待推理的文本列表

        Returns
        -------
        List[List[float]]
            每条文本对应的 logits 向量
        """
        if not texts:
            return []

        bs = self._config.batch_size
        # 无需拆分
        if len(texts) <= bs:
            return await self._predict_batch(texts)

        # 拆分为多个 batch 并发请求
        batches = [texts[i: i + bs] for i in range(0, len(texts), bs)]
        tasks = [self._predict_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        # 合并结果
        all_logits: List[List[float]] = []
        for batch_logits in results:
            all_logits.extend(batch_logits)
        return all_logits

    async def _predict_batch(self, texts: List[str]) -> List[List[float]]:
        """单批次推理请求"""
        resp = await self._request_with_retry(
            "POST",
            "/predict",
            json={"texts": texts},
        )
        data = resp.json()
        return data["logits"]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """数值稳定的 softmax"""
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    async def classify(self, texts: List[str]) -> List[ClassifyResult]:
        """
        批量分类 —— 对 logits 取 argmax，通过内置 label_map 返回类别名称及置信度

        Parameters
        ----------
        texts : List[str]
            待分类文本

        Returns
        -------
        List[ClassifyResult]
            每条文本的预测类别名称和对应概率
        """
        logits = await self.predict(texts)
        logits_arr = np.array(logits)
        probs = self._softmax(logits_arr)
        indices = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(indices)), indices]

        return [
            ClassifyResult(
                label=self._label_map.get(int(idx), f"UNKNOWN_{idx}"),
                confidence=float(conf),
            )
            for idx, conf in zip(indices, confidences)
        ]

    async def classify_one(self, text: str) -> ClassifyResult:
        """
        单条文本分类的便捷方法

        Parameters
        ----------
        text : str
            待分类文本

        Returns
        -------
        ClassifyResult
            预测类别名称和对应概率
        """
        results = await self.classify([text])
        return results[0]


# ──────────────────────────────────────────────
# 快速验证 (python -m infer.client)
# ──────────────────────────────────────────────
async def _demo():
    config = ClientConfig(
        connect_timeout=3.0,
        read_timeout=10.0,
        max_retries=2,
    )
    async with BertIntentClassifyClient("http://localhost:10001", "label_map.csv", config) as client:
        # 0. 查看加载的类别映射
        print(f"📋 类别映射: {client.label_map}")

        # 1. 健康检查
        ok = await client.health()
        print(f"✅ 服务状态: {'正常' if ok else '不可用'}")
        if not ok:
            return

        # 2. 单条推理
        texts = [
            "你好呀小乐",
            "今天几号?",
            "今天天气怎么样？",
            "调低音量",
            "别动啊",
            "你别笑",
            "你哈哈什么",
            "有病",
            "别过来",
            "滚一边去"
        ]
        for text in texts:
            start = time.monotonic()
            result = await client.classify_one(text)
            print(f"🏷️  {text} → {result.label} ({result.confidence:.2%}), "
                  f"used: {time.monotonic() - start:.3f}s")

        # 3. 批量推理
        texts = ["你好", "再见", "今天天气怎么样", "讲个故事"]
        results = await client.classify(texts)
        for t, r in zip(texts, results):
            print(f"   {t} → {r}")


if __name__ == "__main__":
    asyncio.run(_demo())
