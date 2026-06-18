"""
WorkGuard-VL 记忆管理
===================

本模块实现 Agent 的记忆管理机制。

为什么 Agent 需要记忆？
    LLM 是无状态的——每次调用都是独立的，它不记得之前发生过什么。
    但 Agent 需要"记住"之前的观察和动作，才能做出连贯的决策。

    例如：
    - 第1轮：观察到工位A1有人在睡觉 → 决定查询该工位员工
    - 第2轮：查到员工是张三 → 决定检索公司关于睡觉的制度
    - 第3轮：检索到制度规定"睡觉首次警告" → 决定发送告警

    如果没有记忆，第2轮 LLM 不知道为什么要查员工信息，
    第3轮 LLM 不知道查到了什么、为什么要检索制度。

记忆类型：
    - 短期记忆（Short-term Memory）：
        当前推理轮次中的观察、思考、动作。
        每次推理结束后清空（或存入长期记忆）。
        相当于人的"工作记忆"。

    - 长期记忆（Long-term Memory）：
        跨推理轮次的历史记录。
        包括：历史活动统计、告警记录、行为模式。
        相当于人的"长期记忆"。

    - 情景记忆（Episodic Memory）：
        RAG 检索到的制度文档片段。
        每次推理时按需检索，不需要全部记住。
        相当于人查阅"规章制度手册"。
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================================
# 1. 单条记忆记录
# ============================================================================

@dataclass
class MemoryEntry:
    """
    一条记忆记录。

    属性说明：
        role: 记忆类型
            - "observation": Agent 观察到的信息（如 YOLO 检测结果、工具返回值）
            - "thought": Agent 的推理/思考（如"我需要查一下这个工位的员工"）
            - "action": Agent 执行的动作（如调用 query_employee 工具）
            - "result": 动作的执行结果（如工具返回的员工信息）
            - "context": RAG 检索到的上下文（如公司制度条文）
        content: 记忆内容（文本）
        timestamp: 创建时间戳
        metadata: 额外元数据
    """
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为 dict（用于序列化和 prompt 构建）。"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            **self.metadata,
        }

    def format_for_prompt(self) -> str:
        """
        格式化为 prompt 中的文本行。

        示例输出：
            [观察] 工位A1: 检测到1人，bbox=[687, 163, 849, 383]，活动=睡觉
            [思考] 该工位员工为张三（研发部），当前是工作时间，需要判断是否违纪
            [动作] 调用 search_policies(query="工位上睡觉怎么处理")
            [结果] 找到相关制度：工作时间内在工位上睡觉属于违纪行为，首次警告
        """
        role_labels = {
            "observation": "观察",
            "thought": "思考",
            "action": "动作",
            "result": "结果",
            "context": "参考",
        }
        label = role_labels.get(self.role, self.role)
        return f"[{label}] {self.content}"


# ============================================================================
# 2. 短期记忆（工作记忆）
# ============================================================================

class ShortTermMemory:
    """
    短期记忆：当前推理轮次中的所有记录。

    特点：
        - 有容量限制（max_entries），超过后自动丢弃最旧的记录
        - 用于构建当前推理的上下文
        - 每次推理结束后可以清空或归档

    为什么限制容量？
        - LLM 的上下文窗口有限（如 4096 tokens）
        - 记忆太多会挤占 prompt 空间，影响 LLM 生成质量
        - 限制容量迫使 Agent 保持简洁、聚焦
    """

    def __init__(self, max_entries: int = 20):
        """
        参数说明：
            max_entries: 最大记忆条数（默认 20）
                - 太多：占用 prompt 空间
                - 太少：可能丢失重要上下文
                - 推荐：10-30
        """
        self.max_entries = max_entries
        self._entries: deque[MemoryEntry] = deque(maxlen=max_entries)

    def add(self, role: str, content: str, **metadata: Any) -> None:
        """
        添加一条记忆。

        参数说明：
            role: 记忆类型（observation/thought/action/result/context）
            content: 记忆内容
            **metadata: 额外元数据
        """
        self._entries.append(MemoryEntry(
            role=role,
            content=content,
            metadata=metadata,
        ))

    def add_observation(self, content: str, **metadata: Any) -> None:
        """便捷方法：添加一条观察记录。"""
        self.add("observation", content, **metadata)

    def add_thought(self, content: str, **metadata: Any) -> None:
        """便捷方法：添加一条思考记录。"""
        self.add("thought", content, **metadata)

    def add_action(self, content: str, **metadata: Any) -> None:
        """便捷方法：添加一条动作记录。"""
        self.add("action", content, **metadata)

    def add_result(self, content: str, **metadata: Any) -> None:
        """便捷方法：添加一条结果记录。"""
        self.add("result", content, **metadata)

    def add_context(self, content: str, **metadata: Any) -> None:
        """便捷方法：添加一条 RAG 上下文记录。"""
        self.add("context", content, **metadata)

    def get_entries(self) -> list[MemoryEntry]:
        """获取所有记忆条目。"""
        return list(self._entries)

    def format_for_prompt(self, max_chars: int = 3000) -> str:
        """
        将短期记忆格式化为 prompt 文本。

        参数说明：
            max_chars: 最大字符数（防止超出 LLM 上下文窗口）

        返回值：
            str: 格式化的记忆文本
        """
        lines: list[str] = []
        current_length = 0

        # 从最新的开始（最近的记忆最重要）
        for entry in reversed(self._entries):
            line = entry.format_for_prompt()
            if current_length + len(line) > max_chars:
                break
            lines.append(line)
            current_length += len(line)

        lines.reverse()  # 恢复时间顺序
        return "\n".join(lines)

    def clear(self) -> None:
        """清空短期记忆。"""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# ============================================================================
# 3. 长期记忆（持久化存储）
# ============================================================================

class LongTermMemory:
    """
    长期记忆：跨推理轮次的历史记录。

    特点：
        - 持久化到 JSON 文件，程序重启后仍可读取
        - 记录每个工位的历史活动、告警记录
        - 用于生成趋势分析和行为模式

    存储结构：
        memory.json
        {
            "workstations": {
                "A1": {
                    "total_observations": 150,
                    "activity_distribution": {"工作": 120, "睡觉": 5, "玩手机": 25},
                    "alerts": [...],
                    "last_seen": "2026-06-15T10:30:00"
                }
            }
        }
    """

    def __init__(self, path: str | Path = "data/agent_memory.json"):
        """
        参数说明：
            path: 记忆文件路径
        """
        self.path = Path(path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """从文件加载记忆。"""
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"workstations": {}}

    def save(self) -> None:
        """将记忆保存到文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_workstation_memory(self, workstation_id: str) -> dict[str, Any]:
        """获取指定工位的长期记忆。"""
        ws = self._data.setdefault("workstations", {})
        return ws.setdefault(workstation_id, {
            "total_observations": 0,
            "activity_distribution": {},
            "alerts": [],
            "last_seen": None,
        })

    def record_observation(
        self,
        workstation_id: str,
        activity: str,
        timestamp: str | None = None,
    ) -> None:
        """
        记录一次工位观察。

        参数说明：
            workstation_id: 工位ID
            activity: 检测到的活动
            timestamp: 时间戳（可选，默认当前时间）
        """
        ws = self.get_workstation_memory(workstation_id)
        ws["total_observations"] += 1
        ws["activity_distribution"][activity] = (
            ws["activity_distribution"].get(activity, 0) + 1
        )
        ws["last_seen"] = timestamp or time.strftime("%Y-%m-%dT%H:%M:%S")

    def record_alert(
        self,
        workstation_id: str,
        activity: str,
        severity: str,
        details: str = "",
    ) -> None:
        """记录一次告警。"""
        ws = self.get_workstation_memory(workstation_id)
        ws["alerts"].append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "activity": activity,
            "severity": severity,
            "details": details,
        })

    def get_summary(self, workstation_id: str) -> str:
        """
        生成指定工位的记忆摘要（文本格式）。

        用途：塞入 LLM 的 prompt，让 LLM 了解该工位的历史情况。
        """
        ws = self.get_workstation_memory(workstation_id)
        if ws["total_observations"] == 0:
            return f"工位 {workstation_id}: 暂无历史记录"

        dist = ws["activity_distribution"]
        total = ws["total_observations"]
        dist_str = "、".join(
            f"{act}({cnt}次, {cnt/total*100:.0f}%)"
            for act, cnt in sorted(dist.items(), key=lambda x: -x[1])
        )

        alerts_str = ""
        recent_alerts = ws["alerts"][-3:]  # 最近 3 条告警
        if recent_alerts:
            alert_lines = [
                f"  - {a['timestamp']}: {a['activity']} ({a['severity']})"
                for a in recent_alerts
            ]
            alerts_str = "\n最近告警:\n" + "\n".join(alert_lines)

        return (
            f"工位 {workstation_id} 历史记录:\n"
            f"  总观察次数: {total}\n"
            f"  活动分布: {dist_str}\n"
            f"  最后观察: {ws['last_seen']}"
            f"{alerts_str}"
        )


# ============================================================================
# 4. Agent Memory（统一接口）
# ============================================================================

class AgentMemory:
    """
    Agent 记忆管理器：统一管理短期记忆、长期记忆。

    使用方式：
        >>> memory = AgentMemory()
        >>> memory.short_term.add_observation("工位A1检测到1人，活动=工作")
        >>> memory.short_term.add_thought("需要查询该工位员工信息")
        >>> memory.short_term.add_action("调用 query_employee(A1)")
        >>> memory.short_term.add_result("员工: 张三, 部门: 研发部")
        >>>
        >>> # 构建 prompt 上下文
        >>> context = memory.build_context(workstation_id="A1")
        >>>
        >>> # 推理结束后，归档到长期记忆
        >>> memory.archive(workstation_id="A1", activity="工作")

    参数说明：
        long_term_path: 长期记忆文件路径
        max_short_term: 短期记忆最大条数
    """

    def __init__(
        self,
        long_term_path: str | Path = "data/agent_memory.json",
        max_short_term: int = 20,
    ):
        self.short_term = ShortTermMemory(max_entries=max_short_term)
        self.long_term = LongTermMemory(path=long_term_path)

    def build_context(
        self,
        workstation_id: str | None = None,
        max_chars: int = 3000,
    ) -> str:
        """
        构建 Agent 的记忆上下文（塞入 prompt）。

        包含：
            1. 长期记忆摘要（该工位的历史统计）
            2. 短期记忆（当前推理轮次的观察/思考/动作）

        参数说明：
            workstation_id: 工位ID（可选，用于获取长期记忆）
            max_chars: 最大字符数

        返回值：
            str: 格式化的记忆上下文
        """
        parts: list[str] = []

        # 长期记忆摘要
        if workstation_id:
            lt_summary = self.long_term.get_summary(workstation_id)
            parts.append(f"=== 历史记忆 ===\n{lt_summary}")

        # 短期记忆
        st_text = self.short_term.format_for_prompt(max_chars=max_chars)
        if st_text:
            parts.append(f"=== 当前推理 ===\n{st_text}")

        return "\n\n".join(parts) if parts else "（暂无记忆）"

    def archive(
        self,
        workstation_id: str,
        activity: str,
        save: bool = True,
    ) -> None:
        """
        将当前推理结果归档到长期记忆。

        参数说明：
            workstation_id: 工位ID
            activity: 最终判定的活动
            save: 是否立即保存到文件
        """
        self.long_term.record_observation(workstation_id, activity)
        if save:
            self.long_term.save()

    def reset_short_term(self) -> None:
        """清空短期记忆（开始新的推理轮次时调用）。"""
        self.short_term.clear()
