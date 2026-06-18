"""
WorkGuard-VL ReAct 推理循环
=========================

本模块实现 Agent 的 ReAct (Reasoning + Acting) 推理循环。

什么是 ReAct？
    ReAct 是一种 Agent 推理范式，来自论文 "ReAct: Synergizing Reasoning and
    Acting in Language Models" (Yao et al., 2022)。

    核心思想：让 LLM 交替进行"思考"（Reasoning）和"行动"（Acting），
    而不是只思考不行动（Chain-of-Thought）或只行动不思考。

    推理过程：
    ┌─────────────────────────────────────────────────────────┐
    │ 观察: 工位A1有人，bbox=[687,163,849,383]，活动=睡觉      │
    │                                                         │
    │ 思考: 我需要查一下这个工位的员工是谁，然后判断是否违纪      │
    │ 动作: ACTION: query_employee(workstation_id="A1")        │
    │                                                         │
    │ 观察(工具返回): 员工张三，研发部，算法工程师               │
    │ 思考: 当前是工作时间，睡觉属于违纪，需要检索相关制度        │
    │ 动作: ACTION: search_policies(query="工位睡觉处罚")       │
    │                                                         │
    │ 观察(工具返回): 制度规定睡觉首次警告，第二次扣绩效10%       │
    │ 思考: 综合所有信息，可以给出最终判断了                      │
    │ 回答: FINAL_ANSWER: {"activity":"睡觉","compliant":false} │
    └─────────────────────────────────────────────────────────┘

为什么用文本格式而不是 OpenAI function calling？
    1. 通用性：任何 LLM 都能输出文本，不依赖特定 API
    2. 可调试：文本格式人眼可读，方便调试和理解
    3. 兼容性：Ollama 的 tool calling 支持不完全，文本格式更可靠
    4. 教学价值：面试时能讲清楚"如何手写 tool calling parser"
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from agent.memory import AgentMemory
from agent.tools import ToolRegistry


VIOLATION_ACTIVITIES = {"睡觉", "玩手机", "长时间离岗"}


# ============================================================================
# 1. ReAct Prompt 构建
# ============================================================================

# Prompt 模板中使用 {variable} 占位符，运行时替换为实际内容。
# 这是 ReAct 风格的 prompt，明确要求 LLM 按照 观察→思考→动作 的格式输出。

DIRECT_SYSTEM_PROMPT = """你是工位监控分析Agent。你可以看到工位的实时画面，结合工位信息和制度文档进行合规判定。

{tools_description}

===== 输出格式 =====

你必须直接输出最终结果，不要调用工具：
思考: <简要分析>
回答: FINAL_ANSWER: {{"activity":"<主要活动>","transient_activities":["<短暂活动>"],"violation_observed":<true/false>,"confidence":<0-1>,"compliant":<true/false>,"details":"<说明>","alerts_sent":<true/false>}}

===== 活动标签 =====
工作、聊天/讨论、玩手机、睡觉、喝水、离开、其他
如果输入了多张图片，它们按时间顺序排列。请综合整个时间窗口判断主要活动，不要只看最后一张。
短暂出现但不是主要状态的行为放入 transient_activities，例如短暂喝水、短暂转头交流。
请仔细观察画面中人的姿态、手部动作、视线方向来判断，不要默认为"工作"。

===== 合规判定 =====
违纪行为：睡觉、玩手机、长时间离开 → compliant=false, alerts_sent=true
正常行为：工作、聊天/讨论、喝水 → compliant=true, alerts_sent=false
如果只是单帧/短暂疑似玩手机或睡觉但证据不足，violation_observed=false，alerts_sent=false，并在details说明不确定性。"""

REACT_SYSTEM_PROMPT = """你是工位监控分析Agent。你可以看到工位的实时画面，并可以自主调用工具查询员工、排班、历史记录、制度文档和当前时间。

{tools_description}

===== 推理格式 =====

每一步只能输出以下两种格式之一。

需要更多信息时输出：
思考: <为什么需要这个工具>
动作: ACTION: <tool_name>(<key>=<value>, ...)
动作步骤只能包含这两行，不要输出观察、结果、第二个动作或长篇解释。

信息足够时输出：
思考: <简要综合分析>
回答: FINAL_ANSWER: {{"activity":"<主要活动>","transient_activities":["<短暂活动>"],"violation_observed":<true/false>,"confidence":<0-1>,"compliant":<true/false>,"details":"<说明>","alerts_sent":<true/false>}}

===== 工具调用策略 =====
1. 先根据图像判断行为，再按需要调用工具补充上下文。
2. 如果要判断责任人，调用 query_employee。
3. 当前时间通常已在用户上下文中提供；不要为了确认当前时间调用 get_current_time。
4. 如果需要确认员工排班，先 query_employee，再 query_schedule。
5. 如果要查制度依据，调用 search_policies。
6. 如果要参考过往模式，调用 query_activity_history。
7. 不要调用 send_alert。你只能在最终答案中设置 alerts_sent=true，实际告警由程序二次校验后发送。
8. 最多调用一个工具后等待观察结果，不要在同一步写多个 ACTION。
9. 如果一个工具失败或被拦截，必须基于已有观察输出 FINAL_ANSWER，不要重试同一工具。

===== 活动标签 =====
工作、聊天/讨论、玩手机、睡觉、喝水、短暂离岗、长时间离岗、下班、离开、其他
如果输入了多张图片，它们按时间顺序排列。请综合整个时间窗口判断主要活动，不要只看最后一张。
短暂出现但不是主要状态的行为放入 transient_activities，例如短暂喝水、短暂转头交流。
如果上下文说明 ROI 内未检测到人员：
- 离岗时间较短，输出 activity="短暂离岗"。
- 离岗时间较长且仍在排班/工作时段内，输出 activity="长时间离岗"。
- 如果已超过排班下班时间或员工已签退，输出 activity="下班"。

===== 合规判定 =====
明确观察到睡觉、玩手机、长时间离岗等异常时，compliant=false，alerts_sent=true。
正常工作、正常讨论、喝水、短暂离岗、下班等行为，compliant=true，alerts_sent=false。
如果只是短暂疑似异常但证据不足，violation_observed=false，alerts_sent=false，并在 details 说明不确定性。"""

REACT_USER_PROMPT = """工位: {workstation_id}
工位ROI区域: {roi_xyxy}
工位状态: {presence}

{rag_context}

{memory_context}

请重点关注ROI区域标注的工位。若有多张图，请按时间顺序综合判断这段时间窗口的主要行为状态（工作/聊天/讨论/玩手机/睡觉/喝水/短暂离岗/长时间离岗/下班/离开/其他），并说明是否出现过短暂异常行为，然后进行合规判定。"""


# ============================================================================
# 2. Action 解析器
# ============================================================================
# LLM 输出的是文本，我们需要从中提取结构化的动作指令。
#
# 解析目标：
#   "动作: ACTION: query_employee(workstation_id="A1")"
#   → 工具名: "query_employee"
#   → 参数: {"workstation_id": "A1"}
#
#   "回答: FINAL_ANSWER: {"activity": "睡觉", ...}"
#   → 最终结果 dict


def parse_action(text: str) -> tuple[str, dict[str, Any]] | None:
    """
    从 LLM 输出中解析 ACTION 指令。

    参数说明：
        text: LLM 的输出文本（可能包含多行，只需要找到 ACTION 行）

    返回值：
        tuple[str, dict] | None:
            成功时返回 (工具名, 参数字典)
            失败时返回 None

    解析逻辑：
        1. 用正则匹配 "ACTION: tool_name(key=value, ...)" 格式
        2. 解析参数字符串，提取 key=value 对
        3. 处理各种值类型（字符串、数字、布尔）
    """
    # 正则匹配 ACTION 行
    # 匹配格式：ACTION: tool_name(param1="val1", param2="val2")
    # (?:[^)]*?)  匹配括号内的任意内容（非贪婪）
    pattern = r'ACTION:\s*(\w+)\((.*?)\)'
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    tool_name = match.group(1)
    params_str = match.group(2).strip()

    # 解析参数
    arguments: dict[str, Any] = {}
    if params_str:
        # 用正则解析 key=value 对
        # 支持：key="string", key=123, key=true
        # 值可能是带引号的字符串，也可能是数字/布尔
        param_pattern = r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[\w.]+)'
        for m in re.finditer(param_pattern, params_str):
            key = m.group(1)
            value = m.group(2)

            # 去掉引号
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            # 尝试转为数字
            elif value.replace(".", "", 1).isdigit():
                value = int(value) if "." not in value else float(value)
            # 布尔值
            elif value.lower() in ("true", "false"):
                value = value.lower() == "true"

            arguments[key] = value

    return tool_name, arguments


def parse_final_answer(text: str) -> dict[str, Any] | None:
    """
    从 LLM 输出中解析 FINAL_ANSWER 指令。

    参数说明：
        text: LLM 的输出文本

    返回值：
        dict | None: 解析出的 JSON dict，失败返回 None
    """
    # 找到 FINAL_ANSWER: 后面的 JSON
    pattern = r'FINAL_ANSWER:\s*(\{.*?\})'
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    json_str = match.group(1)
    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None


def extract_thought(text: str) -> str:
    """
    从 LLM 输出中提取"思考"部分。

    参数说明：
        text: LLM 的输出文本

    返回值：
        str: 提取到的思考内容（去掉"思考:"前缀）
    """
    pattern = r'思考:\s*(.+?)(?=\n动作:|\n回答:|$)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ============================================================================
# 3. ReAct Agent
# ============================================================================

@dataclass
class AgentStep:
    """
    Agent 推理的单步记录。

    属性说明：
        step_number: 步骤编号（从 1 开始）
        thought: LLM 的思考
        action_name: 调用的工具名（如果有）
        action_args: 工具参数（如果有）
        action_result: 工具返回值（如果有）
        final_answer: 最终回答（如果是最后一步）
        raw_output: LLM 的原始输出
        elapsed_s: 本步耗时
    """
    step_number: int
    thought: str = ""
    action_name: str | None = None
    action_args: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None
    final_answer: dict[str, Any] | None = None
    raw_output: str = ""
    elapsed_s: float = 0.0
    should_stop: bool = False


class ReActAgent:
    """
    ReAct 推理 Agent。

    核心循环：
        1. 构建 prompt（系统提示 + 工具描述 + 记忆 + RAG上下文 + 用户输入）
        2. 调用 LLM
        3. 解析 LLM 输出：
           - 如果是 ACTION → 执行工具 → 结果加入记忆 → 回到步骤 1
           - 如果是 FINAL_ANSWER → 返回结果
        4. 如果超过最大步数 → 强制返回兜底结果

    参数说明：
        registry: 工具注册表
        memory: 记忆管理器
        llm_call_fn: LLM 调用函数（接受 messages 列表，返回文本）
        max_steps: 最大推理步数（防止死循环）
            - 太少：可能还没收集够信息就被截断
            - 太多：浪费时间，可能陷入循环
            - 推荐：3-5
    """

    def __init__(
        self,
        registry: ToolRegistry,
        memory: AgentMemory,
        llm_call_fn: Any,
        max_steps: int = 5,
        allow_tools: bool = False,
    ):
        self.registry = registry
        self.memory = memory
        self.llm_call_fn = llm_call_fn
        self.max_steps = max_steps
        self.allow_tools = allow_tools

    def run(
        self,
        workstation_id: str,
        detected_activity: str,
        confidence: float = 0.0,
        bbox: list[float] | None = None,
        presence: str = "occupied",
        rag_context: str = "",
        roi_xyxy: list[int] | None = None,
    ) -> tuple[dict[str, Any], list[AgentStep]]:
        """
        执行一次完整的 ReAct 推理。

        参数说明：
            workstation_id: 工位ID
            detected_activity: YOLO+VL 模型检测到的活动
            confidence: 检测置信度
            bbox: 人体边界框
            presence: 工位状态（occupied/vacant/recently_left）
            rag_context: RAG 检索到的上下文（可选，如果不提供会自动检索）

        返回值：
            tuple[dict, list[AgentStep]]:
                - 最终结果 dict（包含 activity, confidence, compliant, details）
                - 推理步骤列表（用于调试和展示推理过程）
        """
        steps: list[AgentStep] = []
        called_actions: set[str] = set()

        # 重置短期记忆
        self.memory.reset_short_term()

        # 添加初始观察
        self.memory.short_term.add_observation(
            f"工位 {workstation_id}: 检测到活动={detected_activity}，"
            f"置信度={confidence}，bbox={bbox}，状态={presence}"
        )

        # 主循环
        for step_num in range(1, self.max_steps + 1):
            step = self._execute_step(
                step_num=step_num,
                workstation_id=workstation_id,
                detected_activity=detected_activity,
                confidence=confidence,
                bbox=bbox or [],
                presence=presence,
                rag_context=rag_context,
                roi_xyxy=roi_xyxy,
            )
            steps.append(step)

            # 如果得到了最终回答
            if step.final_answer is not None:
                # 归档到长期记忆
                final_activity = step.final_answer.get("activity", detected_activity)
                self.memory.archive(workstation_id, final_activity)
                return step.final_answer, steps

            if step.should_stop and step.action_name is None:
                fallback = self._build_fallback_result(
                    detected_activity,
                    confidence,
                    steps,
                    "Agent 单步输出多个 ACTION，使用规则兜底结果",
                )
                self.memory.archive(workstation_id, fallback["activity"])
                return fallback, steps

            # 如果是动作调用，执行工具
            if step.action_name is not None:
                action_key = json.dumps(
                    {
                        "name": step.action_name,
                        "args": step.action_args or {},
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if not self.allow_tools:
                    result = {
                        "success": False,
                        "error": "当前 Agent 模式不允许工具调用",
                        "result": None,
                    }
                elif step.action_name == "send_alert":
                    result = {
                        "success": False,
                        "error": "send_alert 由程序二次校验后执行，Agent 只能在 FINAL_ANSWER 中设置 alerts_sent",
                        "result": None,
                    }
                elif action_key in called_actions:
                    result = {
                        "success": False,
                        "error": "重复工具调用已被拦截：请基于已有观察给出 FINAL_ANSWER，不要再次调用同一工具和参数",
                        "result": None,
                    }
                    step.should_stop = True
                else:
                    called_actions.add(action_key)
                    result = self.registry.execute(step.action_name, step.action_args or {})
                step.action_result = result

                # 将工具结果加入记忆
                if result["success"]:
                    result_text = json.dumps(result["result"], ensure_ascii=False)
                    # 截断过长的结果
                    if len(result_text) > 500:
                        result_text = result_text[:500] + "..."
                    self.memory.short_term.add_result(
                        f"工具 {step.action_name} 返回: {result_text}"
                    )
                else:
                    self.memory.short_term.add_result(
                        f"工具 {step.action_name} 失败: {result['error']}"
                    )
                if step.should_stop:
                    return self._build_fallback_result(
                        detected_activity,
                        confidence,
                        steps,
                        f"Agent 工具调用被拦截，使用规则兜底结果",
                    ), steps

        # 超过最大步数，返回规则兜底结果，避免把明显违纪活动误标为合规。
        fallback = self._build_fallback_result(
            detected_activity,
            confidence,
            steps,
            f"Agent 推理超过最大步数({self.max_steps})，使用规则兜底结果",
        )
        self.memory.archive(workstation_id, fallback["activity"])
        return fallback, steps

    def _build_fallback_result(
        self,
        detected_activity: str,
        confidence: float,
        steps: list[AgentStep],
        reason: str,
    ) -> dict[str, Any]:
        fallback_activity = self._infer_fallback_activity(detected_activity, steps)
        is_violation = fallback_activity in VIOLATION_ACTIVITIES
        return {
            "activity": fallback_activity,
            "transient_activities": [],
            "violation_observed": is_violation,
            "confidence": max(float(confidence), 0.8 if is_violation else 0.5),
            "compliant": not is_violation,
            "details": f"{reason}：{fallback_activity}，需人工复核后再告警",
            "alerts_sent": False,
        }

    def _infer_fallback_activity(self, detected_activity: str, steps: list[AgentStep]) -> str:
        """Infer a usable activity from partial ReAct outputs when max steps are exhausted."""
        labels = ["工作", "聊天/讨论", "玩手机", "睡觉", "喝水", "短暂离岗", "长时间离岗", "下班", "离开", "其他"]
        text_parts = [detected_activity]
        for step in steps:
            text_parts.append(step.thought)
            text_parts.append(step.raw_output)
        text = "\n".join(part for part in text_parts if part)

        for label in ["玩手机", "睡觉", "长时间离岗"]:
            if label in text:
                return label
        for label in labels:
            if label in text:
                return label
        return detected_activity if detected_activity in labels else "其他"

    def _execute_step(
        self,
        step_num: int,
        workstation_id: str,
        detected_activity: str,
        confidence: float,
        bbox: list[float],
        presence: str,
        rag_context: str,
        roi_xyxy: list[int] | None = None,
    ) -> AgentStep:
        """执行单步推理。"""
        step = AgentStep(step_number=step_num)
        start = time.perf_counter()

        # 构建 prompt
        tools_desc = self.registry.get_tools_description()

        prompt_template = REACT_SYSTEM_PROMPT if self.allow_tools else DIRECT_SYSTEM_PROMPT
        system_prompt = prompt_template.format(
            tools_description=tools_desc,
        )
        memory_context = self.memory.build_context(
            workstation_id=workstation_id,
            max_chars=2500,
        )

        user_prompt = REACT_USER_PROMPT.format(
            workstation_id=workstation_id,
            roi_xyxy=roi_xyxy if roi_xyxy else "unknown",
            presence=presence,
            rag_context=rag_context if rag_context else "",
            memory_context=memory_context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 调用 LLM
        raw_output = self.llm_call_fn(messages)
        step.raw_output = raw_output
        step.elapsed_s = time.perf_counter() - start

        # 解析输出
        step.thought = extract_thought(raw_output)

        # 尝试解析 FINAL_ANSWER
        final_answer = parse_final_answer(raw_output)
        if final_answer is not None:
            step.final_answer = final_answer
            self.memory.short_term.add_thought(step.thought)
            return step

        action_count = raw_output.count("ACTION:")
        if action_count > 1:
            step.thought = step.thought or "模型在单步中输出了多个 ACTION，触发规则兜底"
            step.should_stop = True
            self.memory.short_term.add_thought(step.thought)
            return step

        # 尝试解析 ACTION
        action = parse_action(raw_output)
        if action is not None:
            step.action_name, step.action_args = action
            self.memory.short_term.add_thought(step.thought)
            self.memory.short_term.add_action(
                f"调用 {step.action_name}({json.dumps(step.action_args, ensure_ascii=False)})"
            )
            return step

        # 既没有 FINAL_ANSWER 也没有 ACTION
        # 可能 LLM 输出了纯文本思考，将其作为思考记录并继续
        self.memory.short_term.add_thought(step.thought or raw_output.strip())

        # 如果最后一步还是没结果，返回兜底
        if step_num >= self.max_steps:
            step.final_answer = {
                "activity": "其他",
                "confidence": 0.3,
                "compliant": True,
                "details": "Agent 未能从 LLM 输出中解析出有效结果",
                "alerts_sent": False,
            }

        return step

    def format_steps(self, steps: list[AgentStep]) -> str:
        """
        将推理步骤格式化为可读文本（用于日志和调试）。

        示例输出：
            [Step 1] (0.85s)
              思考: 需要查询工位A1的员工信息
              动作: query_employee(workstation_id="A1")
              结果: {"found": true, "employees": [...]}

            [Step 2] (1.23s)
              思考: 员工张三，当前工作时间，睡觉属于违纪
              回答: {"activity": "睡觉", "compliant": false, ...}
        """
        lines: list[str] = []
        for step in steps:
            lines.append(f"[Step {step.step_number}] ({step.elapsed_s:.2f}s)")
            if step.thought:
                lines.append(f"  思考: {step.thought}")
            if step.action_name:
                args_str = json.dumps(step.action_args, ensure_ascii=False)
                lines.append(f"  动作: {step.action_name}({args_str})")
                if step.action_result:
                    if step.action_result["success"]:
                        result_str = json.dumps(step.action_result["result"], ensure_ascii=False)
                        if len(result_str) > 200:
                            result_str = result_str[:200] + "..."
                        lines.append(f"  结果: {result_str}")
                    else:
                        lines.append(f"  错误: {step.action_result['error']}")
            if step.final_answer is not None:
                answer_str = json.dumps(step.final_answer, ensure_ascii=False)
                lines.append(f"  回答: {answer_str}")
        return "\n".join(lines)
