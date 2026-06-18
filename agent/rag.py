"""
WorkGuard-VL RAG 模块
====================

本模块实现了完整的 RAG (Retrieval-Augmented Generation) 流程。

RAG 的核心思想：
    LLM 的知识是静态的（训练时的数据），无法知道你公司的具体制度。
    RAG 让 LLM 在回答前，先从你的知识库中检索相关文档片段，
    把这些片段塞进 prompt，这样 LLM 就能基于真实资料回答。

完整流程：
    ┌─────────────┐
    │ 1. 加载文档   │  读取 .md / .txt / .json 文件
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 2. 文本分块   │  把长文档切成小段（chunk），每段约 512 字符
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 3. 向量编码   │  用 embedding 模型把每段文字变成一个向量（数字列表）
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 4. 存入索引   │  用 FAISS 建立向量索引，支持快速相似度搜索
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 5. 检索       │  用户问题 → 向量编码 → 在索引中找最相似的 top-k 段
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 6. 组装上下文 │  把检索到的段落拼入 prompt，交给 LLM 生成回答
    └─────────────┘

依赖安装：
    pip install faiss-cpu sentence-transformers

关键概念解释：
    - Embedding（向量编码）：把一段文字映射到一个高维空间中的点。
      语义相近的文字，在空间中的距离也近。
      例如 "工位上睡觉" 和 "在工位睡觉" 的向量非常接近。
    - FAISS：Facebook 开源的向量检索库，支持十亿级别的向量快速搜索。
      IndexFlatIP 表示用内积（Inner Product）做相似度计算。
    - Chunk（分块）：长文档不能直接塞进 LLM（上下文窗口有限），
      所以要切成小段。切分策略直接影响检索质量。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

# ============================================================================
# 1. 文档加载器 (Document Loader)
# ============================================================================
# RAG 的第一步是把原始文档加载进来。
# 不同格式的文件需要不同的解析方式。
# 这里我们支持 .md、.txt、.json 三种格式。

def load_markdown(path: str | Path) -> list[dict[str, Any]]:
    """
    加载 Markdown 文件，按标题（#）切分成多个 section。

    为什么按标题切分？
        - Markdown 的 # 标题天然标记了语义边界
        - 每个 section 通常是一个独立的主题
        - 比盲目按字符数切分更有语义意义

    返回格式：
        [
            {"text": "## 一、工作时间\n\n1. ...", "source": "policies.md", "section": "一、工作时间"},
            {"text": "## 二、考勤打卡规定\n\n1. ...", "source": "policies.md", "section": "二、考勤打卡规定"},
            ...
        ]

    参数说明：
        path: Markdown 文件路径

    返回值：
        list[dict]: 每个 dict 包含 text（文本内容）、source（来源文件）、section（章节标题）
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")

    sections: list[dict[str, Any]] = []
    # 用正则按标题行切分
    # (?m) 表示多行模式，^ 匹配每行开头
    # #{1,3} 匹配 1-3 个 #（即 #、##、### 级标题）
    parts = re.split(r"(?m)^(#{1,3}\s+.+)$", content)

    # parts 的结构：[前言, 标题1, 内容1, 标题2, 内容2, ...]
    # 奇数位是标题，偶数位是内容
    preamble = parts[0].strip()
    if preamble:
        sections.append({
            "text": preamble,
            "source": path.name,
            "section": "前言",
        })

    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # 去掉标题的 # 前缀，只保留标题文字
        section_title = re.sub(r"^#{1,3}\s+", "", header)
        full_text = f"{header}\n\n{body}" if body else header
        sections.append({
            "text": full_text,
            "source": path.name,
            "section": section_title,
        })
        i += 2

    return sections


def load_text(path: str | Path) -> list[dict[str, Any]]:
    """
    加载纯文本文件，整篇作为一个 section。

    参数说明：
        path: 文本文件路径

    返回值：
        list[dict]: 包含一个元素的列表
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return [{"text": text, "source": path.name, "section": "全文"}]


def load_json_as_text(path: str | Path) -> list[dict[str, Any]]:
    """
    加载 JSON 文件，转成可读文本。

    为什么 JSON 也要支持？
        - 很多知识库的原始数据是 JSON 格式（如数据库导出、API 响应）
        - JSON 直接检索效果不好（键值对结构对 embedding 不友好）
        - 转成自然语言文本后，embedding 能更好地理解语义

    参数说明：
        path: JSON 文件路径

    返回值：
        list[dict]: 每条记录转成一个 section
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    sections: list[dict[str, Any]] = [] 

    # 如果是列表，每条记录一个 section
    if isinstance(data, list):
        for i, item in enumerate(data):
            text = json.dumps(item, ensure_ascii=False, indent=2)
            sections.append({
                "text": text,
                "source": path.name,
                "section": f"记录_{i}",
            })
    # 如果是字典，每个顶级 key 一个 section
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                # 列表值：每条记录一个 section
                for i, item in enumerate(value):
                    text = json.dumps(item, ensure_ascii=False, indent=2)
                    sections.append({
                        "text": text,
                        "source": path.name,
                        "section": f"{key}_{i}",
                    })
            else:
                text = json.dumps({key: value}, ensure_ascii=False, indent=2)
                sections.append({
                    "text": text,
                    "source": path.name,
                    "section": key,
                })

    return sections


def load_documents(path: str | Path) -> list[dict[str, Any]]:
    """
    统一文档加载入口：根据文件扩展名选择对应的加载器。

    这是"策略模式"的体现：
        - 每种文件格式有自己的加载策略
        - 调用者不需要关心具体格式，只需要调用 load_documents()

    参数说明：
        path: 文件路径

    返回值：
        list[dict]: 文档片段列表

    异常：
        ValueError: 不支持的文件格式
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".md", ".markdown"):
        return load_markdown(path)
    elif suffix in (".txt", ".text"):
        return load_text(path)
    elif suffix == ".json":
        return load_json_as_text(path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，支持 .md / .txt / .json")


# ============================================================================
# 2. 文本分块器 (Text Splitter)
# ============================================================================
# 文档加载后，每个 section 可能仍然很长（比如"考勤制度"有几千字）。
# 需要进一步切成小块（chunk），每个 chunk 约 512 字符。
#
# 为什么需要分块？
#   1. Embedding 模型有最大长度限制（通常 512 tokens）
#   2. 太长的文本 embedding 质量会下降（信息被"稀释"）
#   3. 检索时需要精确定位到相关段落，而不是整个文档
#
# 分块策略：
#   - chunk_size: 每个 chunk 的最大字符数
#   - chunk_overlap: 相邻 chunk 重叠的字符数
#   - 为什么要重叠？避免在切分边界处丢失上下文
#     例如："员工须在到岗后10分钟内完成打卡签到。未打卡且未补卡者，视为旷工半天。"
#     如果恰好在"签到"和"未打卡"之间切开，两段都不完整。
#     重叠可以保证边界处的信息在两个 chunk 中都出现。


@dataclass
class TextChunk:
    """
    一个文本块（chunk）。

    属性说明：
        text: 文本内容
        source: 来源文件名（如 "policies.md"）
        section: 所属章节（如 "二、考勤打卡规定"）
        chunk_index: 在该 section 中的第几个 chunk（从 0 开始）
        metadata: 额外元数据（可扩展）
    """
    text: str
    source: str
    section: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def split_text_recursive(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> list[str]:
    """
    递归文本分块器（Recursive Character Text Splitter）。

    这是 LangChain 中最常用的分块策略，我们这里手写实现。

    核心思路：
        1. 先尝试用最大的分隔符（如段落分隔符 \\n\\n）切分
        2. 如果切出来的段落仍然太长，用下一级分隔符（如 \\n）继续切
        3. 如果还是太长，用句号、逗号等继续切
        4. 最后实在不行，按字符数硬切

    参数说明：
        text: 要分块的文本
        chunk_size: 每个 chunk 的最大字符数（默认 512）
            - 太大：embedding 质量下降，检索不精确
            - 太小：丢失上下文，信息碎片化
            - 推荐：256-1024，取决于 embedding 模型
        chunk_overlap: 相邻 chunk 的重叠字符数（默认 64）
            - 通常设为 chunk_size 的 10%-20%
            - 太大：chunk 之间重复太多，浪费存储
            - 太小：边界信息可能丢失
        separators: 分隔符列表，从大到小排列
            - 默认：["\\n\\n", "\\n", "。", "；", "，", " "]
            - 优先用大分隔符切分，保证语义完整性

    返回值：
        list[str]: 分块后的文本列表
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", "；", "，", " "]

    # 如果文本已经够短，直接返回
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # 尝试用当前级别的分隔符切分
    sep = separators[0]
    remaining_separators = separators[1:]

    parts = text.split(sep)

    chunks: list[str] = []
    current_chunk = ""

    for part in parts:
        # 如果加上这个 part 后不超过 chunk_size，就继续拼接
        candidate = current_chunk + sep + part if current_chunk else part

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # 当前 chunk 已经够了，保存它
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # 如果这个 part 本身太长，用下一级分隔符递归切分
            if len(part) > chunk_size and remaining_separators:
                sub_chunks = split_text_recursive(
                    part, chunk_size, chunk_overlap, remaining_separators
                )
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = part

    # 处理最后一个 chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # 添加重叠（overlap）
    # 重叠的目的：让相邻 chunk 有公共部分，避免边界处信息丢失
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks: list[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # 从前一个 chunk 的末尾取 overlap 个字符，拼到当前 chunk 前面
                prev_tail = chunks[i - 1][-chunk_overlap:]
                chunk = prev_tail + chunk
            overlapped_chunks.append(chunk)
        chunks = overlapped_chunks

    return chunks


def chunk_documents(
    sections: list[dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextChunk]:
    """
    对加载后的文档 sections 进行分块。

    参数说明：
        sections: load_documents() 返回的文档片段列表
        chunk_size: 每个 chunk 的最大字符数
        chunk_overlap: 相邻 chunk 的重叠字符数

    返回值：
        list[TextChunk]: 分块后的文本块列表
    """
    all_chunks: list[TextChunk] = []

    for section in sections:
        text = section["text"]
        source = section["source"]
        section_name = section["section"]

        # 用递归分块器切分
        text_chunks = split_text_recursive(text, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(text_chunks):
            all_chunks.append(TextChunk(
                text=chunk_text,
                source=source,
                section=section_name,
                chunk_index=i,
            ))

    return all_chunks


# ============================================================================
# 3. Embedding 模型（向量编码）
# ============================================================================
# Embedding 是 RAG 的核心组件之一。
#
# 什么是 Embedding？
#   把一段文字变成一个固定长度的数字向量（如 384 维）。
#   语义相近的文字，向量之间的距离（余弦相似度）接近 1。
#   语义无关的文字，向量之间的距离接近 0。
#
# 为什么用 sentence-transformers？
#   - 专门为句子/段落级别的 embedding 优化
#   - all-MiniLM-L6-v1 模型只有 ~80MB，384 维，速度快
#   - 支持中英文，适合你的工位监控行业场景
#
# 注意：首次运行会自动从 HuggingFace 下载模型（~80MB）。
#       如果网络不好，可以提前下载到本地，改 model_name 为本地路径。


class EmbeddingModel:
    """
    Embedding 模型封装类。

    使用 SentenceTransformer 加载预训练模型，
    提供 encode() 方法将文本列表转为向量矩阵。

    关键参数说明：
        model_name: HuggingFace 模型名称或本地路径
            - "all-MiniLM-L6-v1": 384 维，~80MB，速度快，推荐
            - "paraphrase-multilingual-MiniLM-L12-v2": 384 维，~470MB，多语言效果更好
            - 也可以用本地路径，如 r"D:\models\all-MiniLM-L6-v1"
        device: 运行设备
            - "cuda": 用 GPU 加速（需要 CUDA）
            - "cpu": 用 CPU（较慢但无 GPU 要求）
            - "auto": 自动选择（有 GPU 用 GPU，否则用 CPU）
        batch_size: 批处理大小
            - 编码时一次处理多少条文本
            - GPU 显存够大可以设大一些（如 64）
            - CPU 或显存小设小一些（如 16）
    """

    # 类变量：缓存已加载的模型，避免重复加载
    # SentenceTransformer 加载一次需要几秒，缓存后后续调用很快
    _models: dict[str, Any] = {}

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v1",
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        # 延迟导入：只在实际使用时才 import
        # 这样不使用 RAG 功能时不需要安装 sentence-transformers
        from sentence_transformers import SentenceTransformer

        # 选择设备
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # 加载模型（带缓存）
        cache_key = f"{model_name}_{device}"
        if cache_key not in EmbeddingModel._models:
            print(f"[RAG] 加载 Embedding 模型: {model_name} (device={device})")
            EmbeddingModel._models[cache_key] = SentenceTransformer(
                model_name, device=device
            )
            print(f"[RAG] Embedding 模型加载完成")
        self.model = EmbeddingModel._models[cache_key]

    @property
    def dimension(self) -> int:
        """返回 embedding 向量的维度。"""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        将文本列表编码为向量矩阵。

        参数说明：
            texts: 要编码的文本列表，如 ["工位上睡觉", "工作时间内玩手机"]

        返回值：
            np.ndarray: shape 为 (len(texts), dimension) 的向量矩阵
                       每一行是一个文本的 embedding 向量

        示例：
            >>> model = EmbeddingModel()
            >>> vectors = model.encode(["工位上睡觉", "在工位睡觉"])
            >>> vectors.shape
            (2, 384)
            >>> # 计算两个向量的余弦相似度
            >>> from numpy.linalg import norm
            >>> cosine_sim = vectors[0] @ vectors[1] / (norm(vectors[0]) * norm(vectors[1]))
            >>> print(f"相似度: {cosine_sim:.4f}")  # 应该接近 1.0
        """
        # normalize_embeddings=True 表示 L2 归一化
        # 归一化后，向量的内积 = 余弦相似度
        # 这样后续用 FAISS IndexFlatIP（内积）搜索就等价于余弦相似度搜索
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,  # 超过 100 条才显示进度条
            normalize_embeddings=True,
        )
        return embeddings


# ============================================================================
# 4. 向量索引（FAISS）
# ============================================================================
# FAISS (Facebook AI Similarity Search) 是最流行的向量检索库。
#
# 核心概念：
#   - Index: 向量索引，存储所有向量，支持快速搜索
#   - IndexFlatIP: 最简单的索引，用内积（Inner Product）计算相似度
#     Flat 表示"暴力搜索"（遍历所有向量），适合中小规模（<100万）
#     IP 表示 Inner Product，配合归一化向量等价于余弦相似度
#   - 对于大规模数据（>100万），可以用 IndexIVFFlat（聚类+搜索）
#     但我们的知识库很小（几十个 chunk），Flat 就够了
#
# 搜索过程：
#   1. 用户问题 → embedding → query_vector (shape: [1, 384])
#   2. FAISS 在索引中找与 query_vector 内积最大的 top-k 个向量
#   3. 返回这些向量的索引和相似度分数


class VectorStore:
    """
    向量存储，基于 FAISS 实现。

    属性说明：
        index: FAISS 索引对象
        chunks: 对应的 TextChunk 列表（索引位置和 chunks 列表位置一一对应）
        dimension: 向量维度
    """

    def __init__(self, dimension: int):
        """
        初始化向量存储。

        参数说明：
            dimension: embedding 向量的维度（如 384）
        """
        self.dimension = dimension
        # IndexFlatIP: 暴力搜索 + 内积相似度
        # 内积 = sum(a[i] * b[i])，对于归一化向量，内积 = 余弦相似度
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[TextChunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[TextChunk]) -> None:
        """
        将向量和对应的文本块添加到索引中。

        参数说明：
            embeddings: shape 为 (n, dimension) 的向量矩阵
            chunks: 对应的 TextChunk 列表，长度必须与 embeddings 行数相同

        注意事项：
            - embeddings 必须是 float32 类型（FAISS 要求）
            - 添加后无法删除单条记录（FAISS 限制），只能重建索引
        """
        # FAISS 要求 float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        在索引中搜索与 query_embedding 最相似的 top-k 个文本块。

        参数说明：
            query_embedding: shape 为 (1, dimension) 或 (dimension,) 的查询向量
            top_k: 返回最相似的 top-k 个结果（默认 3）
                - 太少：可能遗漏关键信息
                - 太多：会把不相关的内容也塞进 prompt，浪费 token
                - 推荐：3-5
            score_threshold: 相似度阈值，低于此分数的结果会被过滤
                - 0.0: 返回所有结果（不过滤）
                - 0.3: 过滤掉相似度低于 0.3 的结果
                - 0.5: 只返回比较相关的结果
                - 对于归一化向量 + 内积，分数范围是 [-1, 1]
                - 实际中，0.3-0.5 通常是比较合理的阈值

        返回值：
            list[dict]: 每个 dict 包含：
                - chunk (TextChunk): 匹配的文本块
                - score (float): 相似度分数（内积，越大越相似）

        示例：
            >>> results = store.search(query_vec, top_k=3, score_threshold=0.3)
            >>> for r in results:
            ...     print(f"分数: {r['score']:.4f}")
            ...     print(f"来源: {r['chunk'].source} / {r['chunk'].section}")
            ...     print(f"内容: {r['chunk'].text[:100]}...")
        """
        if self.index.ntotal == 0:
            return []

        # 确保 query 是 2D 数组
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # FAISS 搜索
        # scores: shape (1, top_k) — 每个查询的 top-k 个相似度分数
        # indices: shape (1, top_k) — 每个查询的 top-k 个结果在索引中的位置
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, actual_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS 用 -1 表示无效结果
                continue
            if score < score_threshold:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
            })

        return results

    def save(self, path: str | Path) -> None:
        """
        将索引保存到磁盘。

        用途：
            - 构建索引很慢（尤其是 embedding 计算），保存后下次直接加载
            - 适合知识库不经常变化的场景

        参数说明：
            path: 保存目录路径，会创建 index.faiss 和 chunks.json 两个文件
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self.index, str(path / "index.faiss"))

        # 保存 chunks 元数据
        chunks_data = [
            {
                "text": c.text,
                "source": c.source,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        (path / "chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, dimension: int) -> "VectorStore":
        """
        从磁盘加载索引。

        参数说明：
            path: 保存目录路径
            dimension: 向量维度（必须与保存时一致）

        返回值：
            VectorStore: 加载后的向量存储实例
        """
        path = Path(path)

        store = cls(dimension)
        store.index = faiss.read_index(str(path / "index.faiss"))

        chunks_data = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        store.chunks = [
            TextChunk(
                text=c["text"],
                source=c["source"],
                section=c["section"],
                chunk_index=c["chunk_index"],
                metadata=c.get("metadata", {}),
            )
            for c in chunks_data
        ]

        return store


# ============================================================================
# 5. RAG Pipeline（完整的 RAG 流程）
# ============================================================================
# 把上面所有组件串起来，提供一个简单的接口。


class RAGPipeline:
    """
    完整的 RAG 流程：加载 → 分块 → 编码 → 存储 → 检索。

    使用方式：
        >>> rag = RAGPipeline()
        >>> rag.add_file("data/policies.md")
        >>> rag.add_file("data/employees.json")
        >>> rag.build_index()
        >>> results = rag.search("工位上睡觉怎么处理？", top_k=3)
        >>> context = rag.format_context(results)
        >>> print(context)

    关键参数说明：
        model_name: Embedding 模型名称
        chunk_size: 文本分块大小（字符数）
        chunk_overlap: 分块重叠大小（字符数）
        device: 运行设备（"auto" / "cuda" / "cpu"）
        index_dir: 索引缓存目录（None 表示不缓存）
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v1",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        device: str = "auto",
        index_dir: str | Path | None = None,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.index_dir = Path(index_dir) if index_dir else None

        self.embedder: EmbeddingModel | None = None
        self.store: VectorStore | None = None
        self.raw_sections: list[dict[str, Any]] = []

    def add_file(self, path: str | Path) -> None:
        """
        添加一个文件到知识库。

        可以多次调用，添加多个文件。
        调用 build_index() 后才会真正构建索引。

        参数说明：
            path: 文件路径（支持 .md / .txt / .json）
        """
        sections = load_documents(path)
        self.raw_sections.extend(sections)
        print(f"[RAG] 加载文件: {Path(path).name} → {len(sections)} 个 section")

    def add_files(self, paths: list[str | Path]) -> None:
        """批量添加多个文件。"""
        for path in paths:
            self.add_file(path)

    def build_index(self) -> None:
        """
        构建向量索引。

        流程：
            1. 检查是否有缓存的索引（index_dir）
            2. 如果有缓存，直接加载
            3. 如果没有，执行：分块 → embedding → 建索引 → 缓存
        """
        # 检查缓存
        if self.index_dir and (self.index_dir / "index.faiss").exists():
            print(f"[RAG] 从缓存加载索引: {self.index_dir}")
            self.embedder = EmbeddingModel(self.model_name, self.device)
            self.store = VectorStore.load(self.index_dir, self.embedder.dimension)
            print(f"[RAG] 索引加载完成: {self.store.index.ntotal} 个向量")
            return

        # 初始化 embedding 模型
        self.embedder = EmbeddingModel(self.model_name, self.device)

        # 分块
        chunks = chunk_documents(
            self.raw_sections, self.chunk_size, self.chunk_overlap
        )
        print(f"[RAG] 文本分块完成: {len(chunks)} 个 chunk")

        # Embedding 编码
        texts = [c.text for c in chunks]
        print(f"[RAG] 开始 embedding 编码...")
        embeddings = self.embedder.encode(texts)
        print(f"[RAG] 编码完成: shape={embeddings.shape}")

        # 建立 FAISS 索引
        self.store = VectorStore(self.embedder.dimension)
        self.store.add(embeddings, chunks)
        print(f"[RAG] 索引构建完成: {self.store.index.ntotal} 个向量")

        # 缓存到磁盘
        if self.index_dir:
            self.store.save(self.index_dir)
            print(f"[RAG] 索引已缓存到: {self.index_dir}")

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        检索与 query 最相关的文档片段。

        参数说明：
            query: 用户问题（自然语言），如 "工位上睡觉怎么处理？"
            top_k: 返回最相似的 top-k 个结果
            score_threshold: 相似度阈值

        返回值：
            list[dict]: 检索结果列表
        """
        if self.store is None or self.embedder is None:
            raise RuntimeError("请先调用 build_index() 构建索引")

        # 问题 → 向量
        query_vec = self.embedder.encode([query])[0]

        # 向量检索
        results = self.store.search(query_vec, top_k, score_threshold)
        return results

    def format_context(
        self,
        results: list[dict[str, Any]],
        max_length: int = 2000,
    ) -> str:
        """
        将检索结果格式化为 LLM 可用的上下文文本。

        为什么需要格式化？
            - 检索结果是结构化的 dict，LLM 需要的是自然语言文本
            - 需要标注来源，方便 LLM 引用和用户验证
            - 需要控制总长度，避免超出 LLM 的上下文窗口

        参数说明：
            results: search() 的返回值
            max_length: 上下文最大字符数（默认 2000）
                - 太大：占用 prompt 空间，影响 LLM 生成质量
                - 太小：信息不足，LLM 无法基于上下文回答

        返回值：
            str: 格式化的上下文文本

        示例输出：
            ---[来源: policies.md / 三、工位行为规范 | 相关度: 0.85]---
            3. 工作时间内在工位上睡觉属于违纪行为，首次警告，第二次扣除当月绩效 10%。

            ---[来源: policies.md / 四、异常行为处理 | 相关度: 0.72]---
            1. 系统监测到工位无人超过 30 分钟，将自动发送提醒通知给员工本人。
        """
        if not results:
            return "（未找到相关文档）"

        parts: list[str] = []
        current_length = 0

        for i, result in enumerate(results):
            chunk = result["chunk"]
            score = result["score"]

            entry = (
                f"---[来源: {chunk.source} / {chunk.section} | "
                f"相关度: {score:.2f}]---\n"
                f"{chunk.text}"
            )

            if current_length + len(entry) > max_length and parts:
                break

            parts.append(entry)
            current_length += len(entry)

        return "\n\n".join(parts)

    def query(
        self,
        question: str,
        top_k: int = 3,
        score_threshold: float = 0.0,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        一站式检索：输入问题，返回格式化的上下文和原始结果。

        参数说明：
            question: 用户问题
            top_k: 返回结果数
            score_threshold: 相似度阈值

        返回值：
            tuple[str, list[dict]]:
                - 格式化的上下文文本（可直接塞入 prompt）
                - 原始检索结果列表
        """
        results = self.search(question, top_k, score_threshold)
        context = self.format_context(results)
        return context, results


# ============================================================================
# 6. 便捷函数
# ============================================================================

def create_rag(
    knowledge_dir: str | Path,
    model_name: str = "all-MiniLM-L6-v1",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    device: str = "auto",
    index_dir: str | Path | None = None,
) -> RAGPipeline:
    """
    快速创建 RAG Pipeline 的便捷函数。

    会自动扫描 knowledge_dir 下的所有 .md / .txt / .json 文件，
    构建索引并返回可用的 RAGPipeline 实例。

    参数说明：
        knowledge_dir: 知识库目录路径
        model_name: Embedding 模型名称
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        device: 运行设备
        index_dir: 索引缓存目录

    返回值：
        RAGPipeline: 构建完成的 RAG 实例

    使用示例：
        >>> rag = create_rag("data/")
        >>> context, results = rag.query("工位上睡觉怎么处理？")
        >>> print(context)
    """
    knowledge_dir = Path(knowledge_dir)

    if index_dir is None:
        index_dir = knowledge_dir / ".rag_index"

    rag = RAGPipeline(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device,
        index_dir=index_dir,
    )

    # 扫描目录下所有支持的文件
    supported_extensions = {".md", ".markdown", ".txt", ".text", ".json"}
    files = sorted(
        f for f in knowledge_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    )

    if not files:
        print(f"[RAG] 警告: {knowledge_dir} 下没有找到支持的文件")
        return rag

    rag.add_files(files)
    rag.build_index()

    return rag
