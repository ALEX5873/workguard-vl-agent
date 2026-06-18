import argparse
import base64
import json
import re
import time
from pathlib import Path
from typing import Any

from ollama import chat


DEFAULT_MODEL = "qwen3.5:2b"


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def build_workstation_vl_activity_prompt(
    workstation_id: str,
    bbox_xyxy: list[float],
    roi_xyxy: list[int] | None,
    last_state: str,
    last_activity: str,
) -> str:
    roi_text = "unknown" if roi_xyxy is None else str([int(v) for v in roi_xyxy])
    bbox_text = str([round(float(v), 1) for v in bbox_xyxy])
    return f"""
你是一个办公室工位行为分析助手。请只分析图片中指定工位内的人，不要分析无关区域。

工位ID: {workstation_id}
工位ROI坐标: {roi_text}
YOLO检测到的人体bbox坐标: {bbox_text}
上一次稳定状态: {last_state}
上一次行为: {last_activity}

请根据完整图片和bbox附近的上下文判断该人员当前行为。只能从以下标签中选择一个：
工作、聊天/讨论、玩手机、睡觉、喝水、短暂离岗、长时间离岗、下班、离开、其他

判断要求：
1. 重点关注bbox内的人，结合桌面、电脑、椅子、手部物品等上下文综合判断。
2. 人最常见的状态是"工作"，只有明确证据支持其他行为时才选择其他标签。
3. 如果看不清或证据不足，activity填"其他"，confidence不要超过0.5。
4. 不要输出思考过程，不要输出额外说明。
5. 严格输出JSON：
{{"activity":"工作","confidence":0.90,"details":"坐在工位前面对电脑屏幕"}}
""".strip()


def ask_ollama_for_workstation_vl_activity(
    image_base64: str,
    workstation_id: str,
    bbox_xyxy: list[float],
    roi_xyxy: list[int] | None = None,
    last_state: str = "unknown",
    last_activity: str = "unknown",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
) -> tuple[dict[str, Any], float, str]:
    """Ask a multimodal Ollama model to classify one workstation frame."""
    prompt = build_workstation_vl_activity_prompt(
        workstation_id=workstation_id,
        bbox_xyxy=bbox_xyxy,
        roi_xyxy=roi_xyxy,
        last_state=last_state,
        last_activity=last_activity,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个谨慎的多模态工位行为识别助手。你只能根据图片证据回答，必须输出JSON。\n"
                "\n"
                "每种行为的判断标准（按可能性从高到低排列，优先匹配最符合的）：\n"
                "1. 工作：人坐在工位前，面向电脑屏幕或键盘，这是最常见的状态。\n"
                "2. 聊天/讨论：人面朝其他人说话，或多人围在一起交流。\n"
                "3. 喝水：人手持杯子/水瓶到嘴边。\n"
                "4. 短暂离岗：工位上暂时无人，离开时间较短。\n"
                "5. 长时间离岗：工位上长时间无人，且仍处于应在岗时段。\n"
                "6. 下班：工位上无人，且已超过排班下班时间或员工已签退。\n"
                "7. 离开：人正在起身、走动或离开工位。\n"
                "8. 睡觉：人趴在桌上、头靠在手臂上、眼睛闭合。\n"
                "9. 玩手机：人手持手机并低头注视屏幕。\n"
                "10. 其他：确实无法匹配以上任何一种时才用此标签。\n"
                "\n"
                "注意：请根据画面整体判断最符合的行为，不要只关注某个局部细节。\n"
            ),
        },
        {
            "role": "user",
            "content": prompt,
            "images": [image_base64],
        },
    ]
    options = {
        "num_ctx": 4096,
        "temperature": temperature,
        "top_p": 0.8,
        "num_predict": 256,
    }

    start_time = time.perf_counter()
    try:
        response = chat(model=model, messages=messages, options=options, think=False)
    except TypeError:
        response = chat(model=model, messages=messages, options=options)
    elapsed_s = time.perf_counter() - start_time

    raw_text = extract_response_text(response)
    parsed = extract_json_object(raw_text) or {}
    activity = parsed.get("activity")
    if activity not in {"工作", "聊天/讨论", "玩手机", "睡觉", "喝水", "短暂离岗", "长时间离岗", "下班", "离开", "其他"}:
        activity = "其他"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    result = {
        "activity": activity,
        "confidence": max(0.0, min(confidence, 1.0)),
        "details": str(parsed.get("details", "")).strip(),
    }
    return result, elapsed_s, raw_text


def build_workstation_vl_report_prompt(report: dict[str, Any]) -> str:
    facts = json.dumps(report, ensure_ascii=False, indent=2)
    return f"""
你是一个办公室工位状态日报助手。请根据结构化日志总结指定工位的上班情况。
只允许使用日志中的事实，不要编造未出现的时间、人员或行为。
禁止只回复“收到”“好的”“OK”。必须输出完整报告。

请严格按以下格式输出，至少 4 行：
总体在岗情况：<一句话>
主要行为分布：<列出 activity_counts 中的行为和次数>
离开/异常行为情况：<说明离开、玩手机、睡觉等异常是否出现>
一句话结论：<一句话>

结构化日志：
{facts}
""".strip()


def build_deterministic_workstation_vl_report(report: dict[str, Any]) -> str:
    current = report.get("current", {})
    counts = report.get("activity_counts", {})
    events = report.get("events", [])
    samples = report.get("samples", [])
    workstation_id = report.get("workstation_id", "unknown")
    count_text = "、".join(f"{key}{value}次" for key, value in counts.items()) or "暂无样本"
    abnormal = [
        sample for sample in samples
        if sample.get("activity") in {"玩手机", "睡觉", "长时间离岗", "离开"}
        or sample.get("stable_activity") in {"玩手机", "睡觉", "长时间离岗", "离开"}
    ]
    abnormal_text = (
        f"发现 {len(abnormal)} 条离开/异常相关样本。"
        if abnormal
        else "未发现离开、玩手机、睡觉等异常样本。"
    )
    return (
        f"总体在岗情况：工位 {workstation_id} 当前状态为 "
        f"{current.get('presence', 'unknown')}，当前行为为 {current.get('activity', '其他')}。\n"
        f"主要行为分布：{count_text}。\n"
        f"离开/异常行为情况：{abnormal_text} 共记录 {len(events)} 个状态事件。\n"
        f"一句话结论：本报告基于结构化日志自动生成，未使用日志外信息。"
    )


def ask_ollama_for_workstation_vl_report(
    report: dict[str, Any],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> tuple[str, float]:
    prompt = build_workstation_vl_report_prompt(report)
    messages = [
        {
            "role": "system",
            "content": "你是谨慎的工位状态总结助手，只根据结构化日志总结。",
        },
        {"role": "user", "content": prompt},
    ]
    options = {
        "num_ctx": 4096,
        "temperature": temperature,
        "top_p": 0.9,
        "num_predict": 300,
    }
    start_time = time.perf_counter()
    try:
        response = chat(model=model, messages=messages, options=options, think=False)
    except TypeError:
        response = chat(model=model, messages=messages, options=options)
    elapsed_s = time.perf_counter() - start_time
    summary = extract_response_text(response).strip()
    if summary in {"", "收到", "好的", "OK", "ok"} or len(summary) < 20:
        summary = build_deterministic_workstation_vl_report(report)
    return summary, elapsed_s


def build_scene_analysis_prompt(scene: dict[str, Any], user_question: str) -> str:
    scene_json = json.dumps(scene, ensure_ascii=False, indent=2)
    return f"""
你是一个室内人员动态安全风险监测 Agent。

你会收到由 ZED2i 深度相机、YOLO11 目标检测与跟踪、深度距离估计、
轨迹分析算法生成的 scene JSON。

请严格根据 scene JSON 回答，不要编造画面中没有的信息。

你的任务：
1. 判断当前是否存在人员碰撞、绊倒或接近障碍物的动态风险。
2. 如果存在风险，说明涉及哪个 person track_id、哪个障碍物、预计多久后接近、预测最近距离是多少。
3. 如果没有动态风险，但存在 static_interaction、near_static_object 等关系，要说明这不是动态碰撞风险。
4. 给出简洁、可执行的安全建议。
5. 如果 scene JSON 信息不足，请明确说明“不确定”，不要猜测。
6. 请直接输出最终回答，不要输出 Thinking Process、推理过程、分析步骤或草稿。

请按下面格式输出：
结论：...
依据：...
建议：...

scene JSON:
{scene_json}

用户问题:
{user_question}
""".strip()


def build_seat_summary_prompt(report: dict[str, Any], user_question: str) -> str:
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    return f"""
你是一个图书馆/自习室座位占用巡检 Agent。

你会收到由 ZED2i 深度相机、YOLO11 目标检测、人员跟踪和座位状态机生成的
seat occupancy report JSON。

请严格根据 JSON 总结，不要编造没有出现的座位、人员或物品。

你的任务：
1. 总结当前座位使用情况，包括正常使用、空闲、临时离开、疑似占座。
2. 如果存在疑似占座，请指出 seat_id、持续时间和证据物品。
3. 如果没有明显问题，请说明当前整体状态正常。
4. 给出简洁的管理员处理建议。
5. 请直接输出最终回答，不要输出 Thinking Process、推理过程或草稿。

请按下面格式输出：
概况：...
问题：...
建议：...

seat occupancy report JSON:
{report_json}

用户问题:
{user_question}
""".strip()


def format_workstation_facts(report: dict[str, Any]) -> str:
    workstations = report.get("workstations", [])
    events = report.get("events", [])
    counts = report.get("summary_counts", {})

    occupied = [item for item in workstations if item.get("status") == "occupied"]
    recently_left = [item for item in workstations if item.get("status") == "recently_left"]
    vacant = [item for item in workstations if item.get("status") == "vacant"]

    def format_items(items: list[dict[str, Any]]) -> str:
        if not items:
            return "无"
        lines = []
        for item in items:
            parts = [
                f'id={item.get("workstation_id")}',
                f'status={item.get("status")}',
                f'duration={item.get("status_duration_s")}s',
            ]
            if item.get("current_person_ids"):
                parts.append(f'person_ids={item.get("current_person_ids")}')
            if item.get("last_person_absent_s") is not None:
                parts.append(f'absent={item.get("last_person_absent_s")}s')
            lines.append("- " + ", ".join(parts))
        return "\n".join(lines)

    recent_event_lines = []
    for event in events[-5:]:
        recent_event_lines.append(
            "- "
            + ", ".join(
                [
                    f'workstation_id={event.get("workstation_id")}',
                    f'event={event.get("event_type")}',
                    f'new_status={event.get("new_status")}',
                ]
            )
        )

    if not recent_event_lines:
        recent_events = "无"
    else:
        recent_events = "\n".join(recent_event_lines)

    return f"""
场景: {report.get("scene")}
时间窗口: {report.get("time_window_s")} 秒
工位总数: {len(workstations)}
统计: occupied={counts.get("occupied", 0)}, recently_left={counts.get("recently_left", 0)}, vacant={counts.get("vacant", 0)}

有人工位:
{format_items(occupied)}

刚离开工位:
{format_items(recently_left)}

空闲工位:
{format_items(vacant)}

最近状态变化:
{recent_events}
""".strip()


def build_deterministic_workstation_summary(report: dict[str, Any]) -> str:
    workstations = report.get("workstations", [])
    counts = report.get("summary_counts", {})
    events = report.get("events", [])

    occupied = [item for item in workstations if item.get("status") == "occupied"]
    recently_left = [item for item in workstations if item.get("status") == "recently_left"]
    vacant = [item for item in workstations if item.get("status") == "vacant"]

    def ids(items: list[dict[str, Any]]) -> str:
        if not items:
            return "无"
        return "、".join(str(item.get("workstation_id")) for item in items)

    status_parts = [
        f"有人：{ids(occupied)}",
        f"刚离开：{ids(recently_left)}",
        f"空闲：{ids(vacant)}",
    ]

    change_parts = []
    for event in events[-3:]:
        workstation_id = event.get("workstation_id")
        new_status = event.get("new_status")
        if workstation_id and new_status:
            change_parts.append(f"{workstation_id} 变为 {new_status}")

    if not change_parts:
        changes = "最近时间窗口内没有新的工位状态变化。"
    else:
        changes = "；".join(change_parts) + "。"

    return (
        f"概况：当前监测到 {len(workstations)} 个工位，"
        f"其中 {counts.get('occupied', 0)} 个有人、"
        f"{counts.get('recently_left', 0)} 个刚离开、"
        f"{counts.get('vacant', 0)} 个空闲。\n"
        f"工位状态：{'；'.join(status_parts)}。\n"
        f"变化：{changes}"
    )


def build_workstation_summary_prompt(report: dict[str, Any], user_question: str) -> str:
    facts = format_workstation_facts(report)
    return f"""
你是一个办公室/实验室工位在岗状态巡检 Agent。

你会收到由固定摄像头、YOLO11 人员检测、人员跟踪和工位 ROI 状态机生成的工位状态事实。

请严格根据事实总结，不要编造没有出现的工位或人员。

你的任务：
1. 总结当前每个工位是否有人。
2. 如果有人刚离开，请说明对应 workstation_id 和离开时长。
3. 如果工位空闲，请说明哪些工位当前无人。
4. 总结最近时间窗口内工位使用变化。
5. 只能提及事实中出现的 workstation_id，不能举例、不能写“如 A1/B2/C3”等不存在的工位。
6. 不要输出 JSON 原文，不要输出字段名列表，不要输出 Thinking Process、推理过程或草稿。

请按下面格式输出：
概况：...
工位状态：...
变化：...

工位状态事实:
{facts}

用户问题:
{user_question}
""".strip()


def extract_response_text(response: Any) -> str:
    """Handle different Ollama response shapes and thinking-model outputs."""
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")

    if message is None:
        return ""

    if isinstance(message, dict):
        content = message.get("content") or ""
        thinking = message.get("thinking") or ""
    else:
        content = getattr(message, "content", "") or ""
        thinking = getattr(message, "thinking", "") or ""

    if content.strip():
        return content

    if thinking.strip():
        return f"[模型只返回了 thinking 字段，未返回最终回答]\n{thinking}"

    return ""


def ask_ollama(
    scene: dict[str, Any],
    user_question: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    return_raw: bool = False,
) -> tuple[str, float] | tuple[str, float, Any]:
    prompt = build_scene_analysis_prompt(scene, user_question)

    start_time = time.perf_counter()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个谨慎、可靠的室内安全风险分析助手。"
                "你只根据结构化数据回答，不根据常识臆测画面。"
                "不要输出思考过程，只输出最终结论。"
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    options = {
        "num_ctx": 4096,
        "temperature": temperature,
        "top_p": 0.9,
        "num_predict": 256,
    }

    try:
        response = chat(
            model=model,
            messages=messages,
            options=options,
            think=False,
        )
    except TypeError:
        response = chat(
            model=model,
            messages=messages,
            options=options,
        )
    end_time = time.perf_counter()

    answer = extract_response_text(response)
    if return_raw:
        return answer, end_time - start_time, response
    return answer, end_time - start_time


def ask_ollama_for_seat_summary(
    report: dict[str, Any],
    user_question: str = "请总结最近一段时间的座位占用情况，并给出管理员建议。",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> tuple[str, float]:
    prompt = build_seat_summary_prompt(report, user_question)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个谨慎、可靠的自习室座位占用巡检助手。"
                "你只根据结构化 JSON 回答，不根据常识臆测。"
                "不要输出思考过程，只输出最终总结。"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    options = {
        "num_ctx": 4096,
        "temperature": temperature,
        "top_p": 0.9,
        "num_predict": 300,
    }

    start_time = time.perf_counter()
    try:
        response = chat(model=model, messages=messages, options=options, think=False)
    except TypeError:
        response = chat(model=model, messages=messages, options=options)
    end_time = time.perf_counter()

    return extract_response_text(response), end_time - start_time


def ask_ollama_for_workstation_summary(
    report: dict[str, Any],
    user_question: str = "请总结最近一段时间办公室/实验室工位是否有人。",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> tuple[str, float]:
    prompt = build_workstation_summary_prompt(report, user_question)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个谨慎、可靠的办公室/实验室工位在岗状态总结助手。"
                "你只根据结构化 JSON 回答，不根据常识臆测。"
                "不要输出占座、抢座、管理员处理占座等图书馆场景内容。"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    options = {
        "num_ctx": 4096,
        "temperature": temperature,
        "top_p": 0.9,
        "num_predict": 120,
        "stop": [
            "workstation occupancy report JSON:",
            "workstation occupancy report",
            "工位状态事实:",
            "场景:",
            "```json",
        ],
    }

    start_time = time.perf_counter()
    try:
        response = chat(model=model, messages=messages, options=options, think=False)
    except TypeError:
        response = chat(model=model, messages=messages, options=options)
    end_time = time.perf_counter()

    answer = extract_response_text(response)
    dirty_markers = [
        "workstation occupancy report JSON:",
        "workstation occupancy report",
        "工位状态事实:",
        "场景:",
        "时间窗口:",
        "```json",
    ]
    for marker in dirty_markers:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    if not answer.strip():
        answer = build_deterministic_workstation_summary(report)

    return answer, end_time - start_time


def load_scene_json(path: str) -> dict[str, Any]:
    scene_path = Path(path)
    with scene_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_demo_scene() -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "camera": "ZED2i",
        "frame_id": 120,
        "objects": [
            {
                "name": "person",
                "track_id": 1,
                "position": "front-center",
                "distance_m": 2.1,
                "camera_x_m": -0.15,
                "camera_z_m": 2.1,
                "motion": "moving_away_from_camera",
                "velocity_mps": {"x": 0.02, "z": 0.35, "speed": 0.351},
                "dynamic_risk": "high",
            },
            {
                "name": "chair",
                "track_id": None,
                "position": "front-center",
                "distance_m": 2.8,
                "camera_x_m": -0.05,
                "camera_z_m": 2.8,
                "static_risk_level": "low",
            },
        ],
        "risk_events": [
            {
                "person_track_id": 1,
                "obstacle_name": "chair",
                "obstacle_track_id": None,
                "risk_level": "high",
                "time_to_closest_s": 1.4,
                "predicted_min_distance_m": 0.42,
                "current_person_obstacle_distance_m": 0.71,
                "current_depth_gap_m": 0.7,
                "relation_state": "approaching",
                "reason": "person trajectory is predicted to pass close to obstacle in camera X-Z space",
            }
        ],
        "has_dynamic_risk": True,
        "yolo_inference_ms": 18.5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Ollama/Qwen scene analysis.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--scene", default=None, help="Path to a scene JSON file.")
    parser.add_argument(
        "--question",
        default="当前是否存在人员碰撞或绊倒风险？如果有，请说明原因和建议。",
        help="User question for the safety agent.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print raw Ollama response when the final answer is empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = load_scene_json(args.scene) if args.scene else build_demo_scene()

    try:
        result = ask_ollama(
            scene=scene,
            user_question=args.question,
            model=args.model,
            return_raw=args.show_raw,
        )
        if args.show_raw:
            answer, elapsed_s, raw_response = result
        else:
            answer, elapsed_s = result
            raw_response = None
    except Exception as exc:
        print("调用 Ollama 失败。请先确认：")
        print(f"1. 已执行：ollama pull {args.model}")
        print("2. Ollama 服务正在运行")
        print("3. 当前 conda 环境已安装：pip install ollama")
        print(f"\n原始错误：{exc}")
        return

    if answer.strip():
        print(answer)
    else:
        print("Ollama 返回成功，但最终回答为空。")
        print("建议用 --show-raw 查看原始响应，确认是否只返回了 thinking 字段。")

    if raw_response is not None and not answer.strip():
        print("\n原始响应：")
        print(raw_response)

    print(f"\nOllama 回答耗时：{elapsed_s:.2f} 秒")


# ============================================================================
# Agent 专用 LLM 调用函数
# ============================================================================

def chat_for_agent(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 512,
    image_base64: str | None = None,
    image_base64s: list[str] | None = None,
) -> str:
    """
    Agent ReAct 推理专用的 LLM 调用函数。

    参数说明：
        messages: Ollama 格式的消息列表
        model: Ollama 模型名称
        temperature: 生成温度
        max_tokens: 最大生成 token 数
        image_base64: 单张工位截图（base64），传入后 Agent 可以看到图片
        image_base64s: 多张工位截图（base64），按时间顺序传入

    返回值：
        str: LLM 的原始文本输出
    """
    images = image_base64s or ([image_base64] if image_base64 else [])
    options = {
        "num_ctx": 8192 if len(images) > 1 else 4096,
        "temperature": temperature,
        "top_p": 0.9,
        "num_predict": max_tokens,
    }

    # 如果有图片，把图片注入到最后一条 user message 中。
    call_messages = list(messages)
    if images:
        for msg in reversed(call_messages):
            if msg.get("role") == "user":
                msg["images"] = images
                break

    try:
        response = chat(model=model, messages=call_messages, options=options, think=False)
    except TypeError:
        response = chat(model=model, messages=call_messages, options=options)

    return extract_response_text(response)


if __name__ == "__main__":
    main()
