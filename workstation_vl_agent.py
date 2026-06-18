import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import base64
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================================
# Tee 输出：同时写到终端和日志文件
# ============================================================================
class TeeWriter:
    """同时写入终端和日志文件的 writer。线程安全。"""
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            self.original.write(text)
            if text.strip():
                self.log_file.write(f"{datetime.now().strftime('%H:%M:%S')} | {text}\n")
                self.log_file.flush()

    def flush(self):
        with self.lock:
            self.original.flush()
            self.log_file.flush()


def setup_logging(log_dir: str = "logs") -> Path:
    """初始化日志，将所有终端输出同时写入日志文件。返回日志文件路径。"""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"workguard_{timestamp}.log"
    fh = open(log_file, "w", encoding="utf-8")
    sys.stdout = TeeWriter(sys.__stdout__, fh)
    sys.stderr = TeeWriter(sys.__stderr__, fh)
    return log_file

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

from llm_client import (
    ask_ollama_for_workstation_vl_activity,
    ask_ollama_for_workstation_vl_report,
    chat_for_agent,
)


DEFAULT_YOLO_MODEL = r"D:\\LLM\nano-vllm-learn-main\yolo11n.engine"
DEFAULT_VL_MODEL = "qwen3.5:9b"
PERSON_CLASS = "person"
ACTIVITY_LABELS = [
    "工作",
    "聊天/讨论",
    "玩手机",
    "睡觉",
    "喝水",
    "短暂离岗",
    "长时间离岗",
    "下班",
    "离开",
    "其他",
]
ACTIVITIES = set(ACTIVITY_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WorkGuard-VL: single-workstation behavior Agent with YOLO + Qwen3.5-VL."
    )
    parser.add_argument("--model", default=DEFAULT_YOLO_MODEL, help="YOLO model path.")
    parser.add_argument(
        "--video",
        default=r".mp4",
        help="Optional offline video. If omitted, ZED2i is used.",
    )
    parser.add_argument(
        "--workstations-config",
        default=r"seats_config.json",
        help="ROI config JSON path. The first ROI is used by default.",
    )
    parser.add_argument("--workstation-id", default=None, help="Target workstation id.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--detect-every", type=int, default=3, help="Run YOLO every N frames.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument(
        "--vl-interval",
        type=float,
        default=30.0,
        help="Seconds between Qwen3.5-VL behavior analyses while occupied.",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=600.0,
        help="Seconds between workstation summary reports. Use a small value for testing.",
    )
    parser.add_argument("--vl-model", default=DEFAULT_VL_MODEL, help="Ollama multimodal model name.")
    parser.add_argument("--disable-vl", action="store_true", help="Disable Qwen3.5-VL analysis.")
    parser.add_argument("--disable-report", action="store_true", help="Disable Qwen report summary.")
    parser.add_argument(
        "--recently-left-s",
        type=float,
        default=60.0,
        help="Seconds without person before stable state becomes 离开.",
    )
    parser.add_argument(
        "--long-left-s",
        type=float,
        default=300.0,
        help="Seconds without person before Agent treats absence as long-leave candidate.",
    )
    parser.add_argument(
        "--vacant-vl-interval",
        type=float,
        default=30.0,
        help="Seconds between Agent analyses while the workstation is recently_left/vacant.",
    )
    parser.add_argument(
        "--alert-cooldown-s",
        type=float,
        default=300.0,
        help="Minimum seconds between repeated alerts for the same workstation/activity.",
    )
    parser.add_argument(
        "--phone-alert-min-frames",
        type=int,
        default=3,
        help="Minimum temporal-window frames required before sending a phone-use alert.",
    )
    parser.add_argument(
        "--activity-confirmations",
        type=int,
        default=2,
        help="Consecutive VL outputs required before changing stable activity.",
    )
    parser.add_argument(
        "--max-vl-image-side",
        type=int,
        default=768,
        help="Resize image before sending to VL model; bbox/ROI are scaled together.",
    )
    parser.add_argument(
        "--vl-window-s",
        type=float,
        default=12.0,
        help="Seconds of recent frames to send to Agent mode as a short temporal window.",
    )
    parser.add_argument(
        "--vl-sample-every-s",
        type=float,
        default=3.0,
        help="Minimum seconds between frames saved into the Agent temporal buffer.",
    )
    parser.add_argument(
        "--vl-max-frames",
        type=int,
        default=5,
        help="Maximum recent frames sent to Agent mode for each reasoning call.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="Playback FPS for offline video.",
    )
    parser.add_argument("--no-video-sync", action="store_true", help="Run video as fast as possible.")
    parser.add_argument(
        "--resolution",
        choices=["HD720", "HD1080", "VGA"],
        default="HD720",
        help="ZED camera resolution.",
    )
    parser.add_argument(
        "--depth-mode",
        choices=["PERFORMANCE", "QUALITY", "NEURAL", "NEURAL_LIGHT"],
        default="PERFORMANCE",
        help="ZED depth mode.",
    )
    parser.add_argument("--no-window", action="store_true", help="Do not show OpenCV preview.")
    # Agent 模式参数
    parser.add_argument("--agent", action="store_true", help="Enable ReAct Agent mode (tool calling + RAG).")
    parser.add_argument(
        "--agent-mode",
        choices=["prefetch", "react"],
        default="prefetch",
        help="Agent reasoning mode: prefetch is faster; react enables autonomous tool calls.",
    )
    parser.add_argument(
        "--employees", default="data/employees.json",
        help="Employee info JSON for Agent mode.",
    )
    parser.add_argument(
        "--schedule", default="data/schedule.json",
        help="Schedule JSON for Agent mode.",
    )
    parser.add_argument(
        "--knowledge-dir", default="data/",
        help="Knowledge base directory for RAG.",
    )
    parser.add_argument(
        "--rag-model", default=r"bge-small-zh-v1.5",
        help="Sentence-Transformer model for RAG embeddings.",
    )
    parser.add_argument(
        "--agent-max-steps", type=int, default=4,
        help="Max ReAct reasoning steps per inference.",
    )
    return parser.parse_args()


def get_resolution(name: str) -> sl.RESOLUTION:
    return {
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "VGA": sl.RESOLUTION.VGA,
    }[name]


def get_depth_mode(name: str) -> sl.DEPTH_MODE:
    return {
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
        "NEURAL": sl.DEPTH_MODE.NEURAL,
        "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,
    }[name]


def open_zed(args: argparse.Namespace) -> sl.Camera:
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = get_resolution(args.resolution)
    init_params.depth_mode = get_depth_mode(args.depth_mode)
    init_params.coordinate_units = sl.UNIT.METER
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED2i open failed: {status}")
    return zed


def sl_mat_to_bgr(image_mat: sl.Mat) -> np.ndarray:
    image = image_mat.get_data()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def load_target_workstation(path: str, workstation_id: str | None) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = json.load(file)
    items = config.get("workstations") or config.get("seats") or []
    if not items:
        raise ValueError(f"No workstation ROI found in {path}")

    selected = None
    if workstation_id is not None:
        for item in items:
            item_id = item.get("workstation_id") or item.get("seat_id")
            if item_id == workstation_id:
                selected = item
                break
        if selected is None:
            raise ValueError(f"workstation id {workstation_id} not found in {path}")
    else:
        selected = items[0]

    return {
        "workstation_id": selected.get("workstation_id") or selected.get("seat_id") or "W1",
        "region_xyxy": [int(v) for v in selected["region_xyxy"]],
    }


def point_in_region(point: tuple[float, float], region_xyxy: list[int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = region_xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def box_area(box_xyxy: list[float]) -> float:
    return max(0.0, box_xyxy[2] - box_xyxy[0]) * max(0.0, box_xyxy[3] - box_xyxy[1])


def get_track_id(box: Any) -> int | None:
    if box.id is None:
        return None
    return int(box.id[0])


def result_to_people_in_roi(model: YOLO, result: Any, roi_xyxy: list[int]) -> list[dict[str, Any]]:
    people = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if not point_in_region(center, roi_xyxy):
            continue
        bbox = [x1, y1, x2, y2]
        people.append(
            {
                "track_id": get_track_id(box),
                "confidence": round(float(box.conf[0]), 4),
                "bbox_xyxy": [round(v, 2) for v in bbox],
                "center_xy": [round(center[0], 2), round(center[1], 2)],
                "area": box_area(bbox),
            }
        )
    people.sort(key=lambda item: item["area"], reverse=True)
    return people


def scale_box(box: list[float] | list[int], scale: float) -> list[float]:
    return [round(float(v) * scale, 2) for v in box]


def prepare_image_for_vl(
    frame: np.ndarray,
    bbox_xyxy: list[float],
    roi_xyxy: list[int],
    max_side: int,
) -> tuple[str, list[float], list[int]]:
    height, width = frame.shape[:2]
    longest = max(height, width)
    scale = 1.0 if longest <= max_side else max_side / longest
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Failed to encode frame for VL model")
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return image_base64, scale_box(bbox_xyxy, scale), [int(v) for v in scale_box(roi_xyxy, scale)]


@dataclass
class VLFrameSample:
    timestamp: float
    image_base64: str
    bbox_xyxy: list[float]
    roi_xyxy: list[int]


class VLFrameBuffer:
    """Keeps a small, time-sampled frame window for Agent temporal reasoning."""

    def __init__(self, window_s: float, sample_every_s: float, max_frames: int) -> None:
        self.window_s = max(0.0, window_s)
        self.sample_every_s = max(0.1, sample_every_s)
        self.max_frames = max(1, max_frames)
        self.samples: list[VLFrameSample] = []
        self.last_sample_time = 0.0

    def maybe_add(
        self,
        frame: np.ndarray,
        bbox_xyxy: list[float],
        roi_xyxy: list[int],
        now: float,
        max_side: int,
    ) -> None:
        if now - self.last_sample_time < self.sample_every_s:
            self._prune(now)
            return

        image_base64, scaled_bbox, scaled_roi = prepare_image_for_vl(
            frame,
            bbox_xyxy,
            roi_xyxy,
            max_side,
        )
        self.samples.append(
            VLFrameSample(
                timestamp=now,
                image_base64=image_base64,
                bbox_xyxy=scaled_bbox,
                roi_xyxy=scaled_roi,
            )
        )
        self.last_sample_time = now
        self._prune(now)

    def get_recent(self, now: float) -> list[VLFrameSample]:
        self._prune(now)
        if not self.samples:
            return []
        return self.samples[-self.max_frames :]

    def _prune(self, now: float) -> None:
        min_time = now - self.window_s
        self.samples = [sample for sample in self.samples if sample.timestamp >= min_time]
        if len(self.samples) > self.max_frames * 2:
            self.samples = self.samples[-self.max_frames * 2 :]


@dataclass
class WorkstationVLState:
    workstation_id: str
    roi_xyxy: list[int]
    stable_presence: str = "unknown"
    stable_activity: str = "其他"
    presence_since: float = field(default_factory=time.time)
    activity_since: float = field(default_factory=time.time)
    last_person_seen: float | None = None
    last_bbox: list[float] | None = None
    last_track_id: int | None = None
    last_vl_time: float = 0.0
    last_vl_result: dict[str, Any] | None = None
    pending_activity: str | None = None
    pending_count: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def update_presence(self, person: dict[str, Any] | None, now: float, recently_left_s: float) -> None:
        if person is not None:
            self.last_person_seen = now
            self.last_bbox = person["bbox_xyxy"]
            self.last_track_id = person.get("track_id")
            self._set_presence("occupied", now)
            return

        if self.last_person_seen is None or now - self.last_person_seen >= recently_left_s:
            self.last_bbox = None
            self.last_track_id = None
            self._set_presence("vacant", now)
            self._set_activity("离开", now)
        else:
            self._set_presence("recently_left", now)
            self._set_activity("短暂离岗", now)

    def apply_vl_result(self, result: dict[str, Any], now: float, confirmations: int, direct: bool = False) -> None:
        activity = result.get("activity", "其他")
        if activity not in ACTIVITIES:
            activity = "其他"

        if direct:
            # Agent 模式：直接更新，跳过确认；相同活动不重复写 activity_changed 事件。
            self._set_activity(activity, now)
            self.pending_activity = None
            self.pending_count = 0
        else:
            if activity == self.stable_activity:
                self.pending_activity = None
                self.pending_count = 0
            elif activity == self.pending_activity:
                self.pending_count += 1
            else:
                self.pending_activity = activity
                self.pending_count = 1

            if self.pending_count >= confirmations:
                self._set_activity(activity, now, force=True)
                self.pending_activity = None
                self.pending_count = 0

        self.last_vl_time = now
        self.last_vl_result = result
        self.samples.append(
            {
                "timestamp": now,
                "presence": self.stable_presence,
                "activity": activity,
                "stable_activity": self.stable_activity,
                "confidence": result.get("confidence"),
                "details": result.get("details", ""),
                "bbox_xyxy": self.last_bbox,
                "track_id": self.last_track_id,
            }
        )

    def _set_presence(self, new_presence: str, now: float) -> None:
        if new_presence == self.stable_presence:
            return
        old = self.stable_presence
        self.stable_presence = new_presence
        self.presence_since = now
        self.events.append(
            {
                "timestamp": now,
                "event_type": "presence_changed",
                "old": old,
                "new": new_presence,
            }
        )

    def _set_activity(self, new_activity: str, now: float, force: bool = False) -> None:
        if new_activity == self.stable_activity and not force:
            return
        old = self.stable_activity
        self.stable_activity = new_activity
        self.activity_since = now
        self.events.append(
            {
                "timestamp": now,
                "event_type": "activity_changed",
                "old": old,
                "new": new_activity,
            }
        )

    def build_report(self, window_s: float | None = None) -> dict[str, Any]:
        now = time.time()
        if window_s is None:
            events = list(self.events)
            samples = list(self.samples)
        else:
            events = [item for item in self.events if now - item["timestamp"] <= window_s]
            samples = [item for item in self.samples if now - item["timestamp"] <= window_s]

        activity_counts: dict[str, int] = {}
        for sample in samples:
            activity = sample.get("stable_activity") or sample.get("activity") or "其他"
            activity_counts[activity] = activity_counts.get(activity, 0) + 1

        return {
            "scene": "single_workstation_vl_monitoring",
            "timestamp": now,
            "window_s": window_s,
            "workstation_id": self.workstation_id,
            "roi_xyxy": self.roi_xyxy,
            "current": {
                "presence": self.stable_presence,
                "presence_duration_s": round(now - self.presence_since, 1),
                "activity": self.stable_activity,
                "activity_duration_s": round(now - self.activity_since, 1),
                "last_bbox": self.last_bbox,
                "last_track_id": self.last_track_id,
                "last_vl_result": self.last_vl_result,
            },
            "activity_counts": activity_counts,
            "events": events,
            "samples": samples[-50:],
        }


class VLActivityWorker:
    def __init__(self, model: str, state: WorkstationVLState, confirmations: int) -> None:
        self.model = model
        self.state = state
        self.confirmations = confirmations
        self.is_running = False
        self.latest_error = ""
        self.latest_elapsed_s = 0.0
        self.elapsed_history_s: list[float] = []
        self.lock = threading.Lock()

    def trigger(self, image_base64: str, bbox_xyxy: list[float], roi_xyxy: list[int]) -> bool:
        with self.lock:
            if self.is_running:
                return False
            self.is_running = True
            self.latest_error = ""
        threading.Thread(
            target=self._run,
            args=(image_base64, bbox_xyxy, roi_xyxy),
            daemon=True,
        ).start()
        return True

    def _run(self, image_base64: str, bbox_xyxy: list[float], roi_xyxy: list[int]) -> None:
        try:
            result, elapsed_s, raw_text = ask_ollama_for_workstation_vl_activity(
                image_base64=image_base64,
                workstation_id=self.state.workstation_id,
                bbox_xyxy=bbox_xyxy,
                roi_xyxy=roi_xyxy,
                last_state=self.state.stable_presence,
                last_activity=self.state.stable_activity,
                model=self.model,
            )
            self.state.apply_vl_result(result, time.time(), self.confirmations)
            with self.lock:
                self.latest_elapsed_s = elapsed_s
                self.elapsed_history_s.append(elapsed_s)
            print("\nQwen3.5-VL behavior analysis:")
            print(json.dumps(result, ensure_ascii=False))
            print(f"VL elapsed: {elapsed_s:.2f}s")
            if raw_text and raw_text.strip() and raw_text.strip()[0] != "{":
                print(f"Raw VL response: {raw_text}")
        except Exception as exc:
            with self.lock:
                self.latest_error = str(exc)
            print(f"\nQwen3.5-VL analysis failed: {exc}")
        finally:
            with self.lock:
                self.is_running = False

    def get_status(self) -> tuple[bool, str, float]:
        with self.lock:
            return self.is_running, self.latest_error, self.latest_elapsed_s


class AgentActivityWorker:
    """
    Agent 模式的活动分析 Worker。

    与 VLActivityWorker 的区别：
        - VLActivityWorker: 直接调用 VL 模式做一次推理，返回 JSON
        - AgentActivityWorker: 运行 ReAct 推理循环，自主决定调用哪些工具

    工作流程：
        1. 接收 YOLO 检测结果（bbox、活动等）
        2. 触发 ReAct Agent 推理
        3. Agent 自主调用工具（查员工、查排班、检索制度、发送告警）
        4. 返回最终判断结果
    """

    def __init__(
        self,
        model: str,
        state: WorkstationVLState,
        confirmations: int,
        registry: Any,
        rag: Any,
        memory: Any,
        max_steps: int = 5,
        agent_mode: str = "prefetch",
        long_left_s: float = 300.0,
        alert_cooldown_s: float = 300.0,
        phone_alert_min_frames: int = 3,
    ) -> None:
        """
        参数说明：
            model: Ollama 模型名称
            state: 工位状态对象
            confirmations: 连续确认次数（与 VL 模式一致）
            registry: 工具注册表
            rag: RAG Pipeline
            memory: 记忆管理器
            max_steps: ReAct 最大推理步数
            agent_mode: prefetch=预取上下文后直接判断；react=自主调用工具
            long_left_s: 长时间离岗判定候选阈值
            alert_cooldown_s: 相同工位/活动重复告警冷却时间
            phone_alert_min_frames: 玩手机告警需要的最少时间窗口帧数
        """
        self.model = model
        self.state = state
        self.confirmations = confirmations
        self.registry = registry
        self.rag = rag
        self.memory = memory
        self.max_steps = max_steps
        self.agent_mode = agent_mode
        self.long_left_s = long_left_s
        self.alert_cooldown_s = alert_cooldown_s
        self.phone_alert_min_frames = max(1, phone_alert_min_frames)
        self.last_alert_times: dict[tuple[str, str], float] = {}
        self.is_running = False
        self.latest_error = ""
        self.latest_elapsed_s = 0.0
        self.latest_steps_text = ""
        self.elapsed_history_s: list[float] = []
        self.lock = threading.Lock()

        # 当前图片序列（每次触发时更新，按时间顺序排列）
        self._current_image_base64s: list[str] = []

        # 创建 LLM 调用函数（闭包，捕获 model 参数和图片）
        def llm_call(messages: list[dict]) -> str:
            return chat_for_agent(
                messages, model=self.model,
                image_base64s=self._current_image_base64s or None,
            )

        # 创建 ReAct Agent
        from agent.react import ReActAgent
        self.agent = ReActAgent(
            registry=self.registry,
            memory=self.memory,
            llm_call_fn=llm_call,
            max_steps=self.max_steps,
            allow_tools=self.agent_mode == "react",
        )

    def trigger(
        self,
        image_base64: str | list[str],
        bbox_xyxy: list[float],
        roi_xyxy: list[int],
        frame_offsets_s: list[float] | None = None,
        no_person: bool = False,
    ) -> bool:
        """触发 Agent 推理（异步）。"""
        with self.lock:
            if self.is_running:
                return False
            self.is_running = True
            self.latest_error = ""
        threading.Thread(
            target=self._run,
            args=(image_base64, bbox_xyxy, roi_xyxy, frame_offsets_s, no_person),
            daemon=True,
        ).start()
        return True

    def _run(
        self,
        image_base64: str | list[str],
        bbox_xyxy: list[float],
        roi_xyxy: list[int],
        frame_offsets_s: list[float] | None = None,
        no_person: bool = False,
    ) -> None:
        try:
            # 保存图片序列，让 Agent 的 LLM 能看到短时间窗口画面。
            if isinstance(image_base64, list):
                self._current_image_base64s = image_base64
            else:
                self._current_image_base64s = [image_base64]
            frame_count = len(self._current_image_base64s)
            frame_offsets_s = frame_offsets_s or [0.0]

            ws_id = self.state.workstation_id
            activity = self.state.stable_activity
            presence = self.state.stable_presence
            absence_s = (
                None
                if self.state.last_person_seen is None
                else max(0.0, time.time() - self.state.last_person_seen)
            )
            current_time = datetime.now()
            current_time_context = (
                "=== 当前时间 ===\n"
                f"当前本地时间: {current_time.isoformat(timespec='seconds')}\n"
                f"日期: {current_time.strftime('%Y-%m-%d')}, "
                f"时间: {current_time.strftime('%H:%M:%S')}, "
                f"星期: {['周一', '周二', '周三', '周四', '周五', '周六', '周日'][current_time.weekday()]}, "
                f"是否工作日: {current_time.weekday() < 5}\n"
                "除非这些时间信息缺失或明显矛盾，不要为了获取当前时间调用 get_current_time。"
            )

            emp_info = ""
            rag_context = ""
            if self.agent_mode == "prefetch":
                # ---- 预取信息：提前调用工具，减少 ReAct 轮次 ----
                emp_result = self.registry.execute("query_employee", {"workstation_id": ws_id})
                if emp_result["success"] and emp_result["result"].get("found"):
                    employees = emp_result["result"]["employees"]
                    emp_lines = [
                        f"  {e['name']}({e['department']}, {e['position']})"
                        for e in employees
                    ]
                    emp_info = f"工位 {ws_id} 员工:\n" + "\n".join(emp_lines)

                    sched_lines = []
                    for e in employees:
                        sched_result = self.registry.execute("query_schedule", {"employee_id": e["employee_id"]})
                        if sched_result["success"] and sched_result["result"].get("found"):
                            s = sched_result["result"]["schedule"]
                            sched_lines.append(f"  {s['name']}: {s['today_status']} ({s['work_hours']})")
                    if sched_lines:
                        emp_info += "\n今日排班:\n" + "\n".join(sched_lines)

                if self.rag is not None:
                    query = f"工位行为规范 {activity} 处罚规定"
                    context, _ = self.rag.query(query, top_k=3)
                    rag_context = f"参考制度文档:\n{context}"
            else:
                rag_context = (
                    "=== 自主工具调用模式 ===\n"
                    "请先观察图像窗口。只有在确实需要员工、排班、制度、历史或当前时间信息时，"
                    "再输出 ACTION 调用一个合适工具。信息足够时直接输出 FINAL_ANSWER。"
                )

            # 把预取信息拼入 rag_context，Agent 就不需要再调工具查了
            frame_context = (
                "=== 时间窗口图像 ===\n"
                f"本次输入包含 {frame_count} 张按时间顺序排列的工位图片。"
                f"相对当前触发时刻的时间偏移约为: {frame_offsets_s} 秒。\n"
                f"当前是否未在 ROI 内检测到人员: {no_person}。\n"
                f"距离上次在 ROI 内看到人员约: "
                f"{round(absence_s, 1) if absence_s is not None else 'unknown'} 秒。\n"
                f"长时间离岗候选阈值: {round(self.long_left_s, 1)} 秒。\n"
                "若未检测到人员且离岗时间低于阈值，优先判断为短暂离岗；"
                "若超过阈值，请结合当前时间和排班判断长时间离岗或下班。\n"
                "不得声称超过本时间窗口之外的持续时长；例如输入只覆盖约12秒，就不要写持续30秒或20分钟。\n"
                "请综合整个时间窗口判断主要行为，并记录是否出现过短暂异常；"
                "不要只根据最后一张图片下结论。"
            )
            frame_context = f"{current_time_context}\n\n{frame_context}"
            rag_context = f"{frame_context}\n\n{rag_context}" if rag_context else frame_context
            if emp_info:
                rag_context = f"=== 工位信息 ===\n{emp_info}\n\n{rag_context}"

            # 运行 ReAct 推理（预取信息后，Agent 可以更快给出结论）
            start = time.perf_counter()
            result, steps = self.agent.run(
                workstation_id=self.state.workstation_id,
                detected_activity=self.state.stable_activity,
                confidence=0.8,
                bbox=bbox_xyxy,
                presence=self.state.stable_presence,
                rag_context=rag_context,
                roi_xyxy=roi_xyxy,
            )
            elapsed_s = time.perf_counter() - start

            # 格式化推理步骤（用于日志）
            steps_text = self.agent.format_steps(steps)

            # 将 Agent 结果转为与 VL 模式兼容的格式
            agent_activity = result.get("activity", "其他")
            # 处理多标签情况（如"工作、聊天/讨论"），提取第一个有效标签
            if agent_activity not in ACTIVITIES:
                for label in ACTIVITY_LABELS:
                    if label in agent_activity:
                        agent_activity = label
                        break
                else:
                    agent_activity = "其他"
            agent_alerts_sent = result.get("alerts_sent", False)

            # 如果 Agent 判定需要告警，自动执行 send_alert
            # 额外校验：只有真正违纪的活动才发告警（防止 2B 模型误判正常行为）
            VIOLATION_ACTIVITIES = {"睡觉", "玩手机", "长时间离岗"}
            should_alert = self._should_send_alert(
                activity=agent_activity,
                result=result,
                frame_count=frame_count,
                absence_s=absence_s,
            ) and agent_activity in VIOLATION_ACTIVITIES
            if should_alert:
                alert_severity = "medium" if agent_activity == "睡觉" else "low"
                self.registry.execute("send_alert", {
                    "workstation_id": ws_id,
                    "activity": agent_activity,
                    "severity": alert_severity,
                    "details": result.get("details", ""),
                })

            vl_result = {
                "activity": agent_activity,
                "confidence": result.get("confidence", 0.5),
                "details": result.get("details", ""),
                "compliant": result.get("compliant", True),
                "alerts_sent": agent_alerts_sent,
                "frame_count": frame_count,
                "frame_offsets_s": frame_offsets_s,
                "transient_activities": result.get("transient_activities", []),
                "violation_observed": result.get("violation_observed", agent_alerts_sent),
            }

            # 应用结果到状态机（Agent 模式直接更新，跳过确认）
            self.state.apply_vl_result(vl_result, time.time(), self.confirmations, direct=True)

            with self.lock:
                self.latest_elapsed_s = elapsed_s
                self.elapsed_history_s.append(elapsed_s)
                self.latest_steps_text = steps_text

            print(f"\n{'=' * 50}")
            print(f"Agent 推理结果 ({self.agent_mode}):")
            print(json.dumps(vl_result, ensure_ascii=False))
            print(f"\n推理过程:")
            print(steps_text)
            print(f"Agent elapsed: {elapsed_s:.2f}s")
            print(f"{'=' * 50}")

        except Exception as exc:
            with self.lock:
                self.latest_error = str(exc)
            print(f"\nAgent analysis failed: {exc}")
        finally:
            self._current_image_base64s = []
            with self.lock:
                self.is_running = False

    def get_status(self) -> tuple[bool, str, float]:
        with self.lock:
            return self.is_running, self.latest_error, self.latest_elapsed_s

    def _should_send_alert(
        self,
        activity: str,
        result: dict[str, Any],
        frame_count: int,
        absence_s: float | None,
    ) -> bool:
        if not result.get("alerts_sent", False):
            return False
        if not result.get("violation_observed", False):
            return False
        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.85:
            return False
        details = str(result.get("details", ""))
        if details.startswith("Agent ") or "规则兜底" in details:
            return False
        if activity == "玩手机" and frame_count < self.phone_alert_min_frames:
            return False
        if activity == "长时间离岗" and (absence_s is None or absence_s < self.long_left_s):
            return False

        key = (self.state.workstation_id, activity)
        now = time.time()
        last = self.last_alert_times.get(key, 0.0)
        if now - last < self.alert_cooldown_s:
            return False
        self.last_alert_times[key] = now
        return True


class ReportWorker:
    def __init__(self, model: str) -> None:
        self.model = model
        self.is_running = False
        self.latest_summary = ""
        self.latest_elapsed_s = 0.0
        self.latest_error = ""
        self.lock = threading.Lock()

    def trigger(self, report: dict[str, Any]) -> bool:
        with self.lock:
            if self.is_running:
                return False
            self.is_running = True
            self.latest_error = ""
        snapshot = json.loads(json.dumps(report, ensure_ascii=False))
        threading.Thread(target=self._run, args=(snapshot,), daemon=True).start()
        return True

    def _run(self, report: dict[str, Any]) -> None:
        try:
            summary, elapsed_s = ask_ollama_for_workstation_vl_report(report, model=self.model)
            with self.lock:
                self.latest_summary = summary
                self.latest_elapsed_s = elapsed_s
            print("\nWorkstation VL report:")
            print(summary)
            print(f"Report elapsed: {elapsed_s:.2f}s")
        except Exception as exc:
            with self.lock:
                self.latest_error = str(exc)
            print(f"\nWorkstation report failed: {exc}")
        finally:
            with self.lock:
                self.is_running = False

    def get_status(self) -> tuple[str, float, str, bool]:
        with self.lock:
            return self.latest_summary, self.latest_elapsed_s, self.latest_error, self.is_running


def format_latency_stats(name: str, values: list[float], unit: str) -> str:
    if not values:
        return f"{name}: no samples"
    arr = np.array(values, dtype=np.float64)
    return (
        f"{name}: samples={len(values)}, mean={arr.mean():.2f}{unit}, "
        f"p50={np.percentile(arr, 50):.2f}{unit}, "
        f"p95={np.percentile(arr, 95):.2f}{unit}, "
        f"min={arr.min():.2f}{unit}, max={arr.max():.2f}{unit}"
    )


def draw_overlay(
    frame: np.ndarray,
    state: WorkstationVLState,
    person: dict[str, Any] | None,
    status_text: str,
) -> np.ndarray:
    x1, y1, x2, y2 = state.roi_xyxy
    color = (0, 255, 0) if state.stable_presence == "occupied" else (180, 180, 180)
    if state.stable_presence == "recently_left":
        color = (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{state.workstation_id}: {state.stable_presence}/{state.stable_activity}"
    cv2.putText(frame, label, (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    if person is not None:
        bx1, by1, bx2, by2 = [int(v) for v in person["bbox_xyxy"]]
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 255), 2)
        track_id = person.get("track_id")
        person_label = f"person {track_id}" if track_id is not None else "person"
        cv2.putText(frame, person_label, (bx1, max(24, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.putText(frame, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


def main() -> None:
    args = parse_args()

    # 初始化日志：所有终端输出同时写入 logs/ 目录
    log_path = setup_logging()
    print(f"[日志] 所有输出同步写入: {log_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"YOLO model not found: {model_path.resolve()}")
        return

    workstation = load_target_workstation(args.workstations_config, args.workstation_id)
    state = WorkstationVLState(
        workstation_id=workstation["workstation_id"],
        roi_xyxy=workstation["region_xyxy"],
    )
    # 根据模式创建 Worker
    vl_worker = None
    if args.agent:
        # Agent 模式：初始化 RAG + 工具 + 记忆 + ReAct Agent
        print("=" * 50)
        print("Agent 模式启动中...")
        print("=" * 50)

        from agent.rag import create_rag
        from agent.tools import create_default_registry
        from agent.memory import AgentMemory

        # 初始化 RAG
        rag = create_rag(
            knowledge_dir=args.knowledge_dir,
            model_name=args.rag_model,
            device="auto",
            index_dir=Path(args.knowledge_dir) / ".rag_index",
        )

        # 初始化工具注册表
        registry = create_default_registry(
            employees_path=args.employees,
            schedule_path=args.schedule,
            rag=rag,
        )
        print(f"[Agent] 已注册 {len(registry)} 个工具: {registry.list_tools()}")

        # 初始化记忆管理器
        memory = AgentMemory()

        # 创建 Agent Worker
        vl_worker = AgentActivityWorker(
            model=args.vl_model,
            state=state,
            confirmations=args.activity_confirmations,
            registry=registry,
            rag=rag,
            memory=memory,
            max_steps=args.agent_max_steps,
            agent_mode=args.agent_mode,
            long_left_s=args.long_left_s,
            alert_cooldown_s=args.alert_cooldown_s,
            phone_alert_min_frames=args.phone_alert_min_frames,
        )
        print(f"[Agent] Agent Worker 初始化完成, mode={args.agent_mode}")
        print("=" * 50)
    elif not args.disable_vl:
        vl_worker = VLActivityWorker(args.vl_model, state, args.activity_confirmations)

    report_worker = None if args.disable_report else ReportWorker(args.vl_model)
    frame_buffer = VLFrameBuffer(
        window_s=args.vl_window_s,
        sample_every_s=args.vl_sample_every_s,
        max_frames=args.vl_max_frames,
    )

    print(f"Target workstation: {state.workstation_id}, roi={state.roi_xyxy}")
    print(f"Loading YOLO model: {model_path.resolve()}")
    model = YOLO(str(model_path))
    print("YOLO model loaded")

    zed = None
    cap = None
    try:
        if args.video:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                print(f"Failed to open video: {args.video}")
                return
            print(f"Offline video opened: {args.video}")
        else:
            zed = open_zed(args)
            print("ZED2i opened")

        image_mat = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        frame_id = 0
        last_person = None
        last_yolo_ms = 0.0
        yolo_latency_ms: list[float] = []
        last_report_time = time.time()
        frame_delay_s = 0.0 if args.no_video_sync else 1.0 / max(args.video_fps, 1.0)

        print("Start WorkGuard-VL. Press q to quit.")
        while True:
            loop_start = time.perf_counter()
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("Video ended")
                    break
            else:
                if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                    continue
                zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                frame = sl_mat_to_bgr(image_mat)

            frame_id += 1
            now = time.time()

            if frame_id % args.detect_every == 0:
                start = time.perf_counter()
                results = model.track(
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    persist=True,
                    tracker=args.tracker,
                    verbose=False,
                )
                last_yolo_ms = (time.perf_counter() - start) * 1000
                yolo_latency_ms.append(last_yolo_ms)
                people = result_to_people_in_roi(model, results[0], state.roi_xyxy)
                last_person = people[0] if people else None
                state.update_presence(last_person, now, args.recently_left_s)

            if (
                isinstance(vl_worker, AgentActivityWorker)
                and last_person is not None
                and state.stable_presence == "occupied"
            ):
                try:
                    frame_buffer.maybe_add(
                        frame,
                        last_person["bbox_xyxy"],
                        state.roi_xyxy,
                        now,
                        args.max_vl_image_side,
                    )
                except Exception as exc:
                    print(f"Failed to buffer Agent frame: {exc}")

            is_agent_worker = isinstance(vl_worker, AgentActivityWorker)
            vacant_agent_due = (
                is_agent_worker
                and last_person is None
                and state.stable_presence in {"recently_left", "vacant"}
                and now - state.last_vl_time >= args.vacant_vl_interval
            )
            should_call_vl = (
                vl_worker is not None
                and (
                    (
                        last_person is not None
                        and state.stable_presence == "occupied"
                        and now - state.last_vl_time >= args.vl_interval
                    )
                    or vacant_agent_due
                )
            )
            if should_call_vl:
                try:
                    if isinstance(vl_worker, AgentActivityWorker):
                        if vacant_agent_due:
                            image_base64, scaled_bbox, scaled_roi = prepare_image_for_vl(
                                frame,
                                state.roi_xyxy,
                                state.roi_xyxy,
                                args.max_vl_image_side,
                            )
                            image_input = [image_base64]
                            frame_offsets_s = [0.0]
                            no_person = True
                        else:
                            samples = frame_buffer.get_recent(now)
                            if samples:
                                image_input = [sample.image_base64 for sample in samples]
                                scaled_bbox = samples[-1].bbox_xyxy
                                scaled_roi = samples[-1].roi_xyxy
                                frame_offsets_s = [round(sample.timestamp - now, 1) for sample in samples]
                            else:
                                image_base64, scaled_bbox, scaled_roi = prepare_image_for_vl(
                                    frame,
                                    last_person["bbox_xyxy"],
                                    state.roi_xyxy,
                                    args.max_vl_image_side,
                                )
                                image_input = [image_base64]
                                frame_offsets_s = [0.0]
                            no_person = False
                        triggered = vl_worker.trigger(
                            image_input,
                            scaled_bbox,
                            scaled_roi,
                            frame_offsets_s=frame_offsets_s,
                            no_person=no_person,
                        )
                    else:
                        image_base64, scaled_bbox, scaled_roi = prepare_image_for_vl(
                            frame,
                            last_person["bbox_xyxy"],
                            state.roi_xyxy,
                            args.max_vl_image_side,
                        )
                        triggered = vl_worker.trigger(image_base64, scaled_bbox, scaled_roi)
                    if triggered:
                        state.last_vl_time = now
                        is_agent = isinstance(vl_worker, AgentActivityWorker)
                        mode = "Agent ReAct" if is_agent else "Qwen3.5-VL"
                        frame_count = len(image_input) if is_agent else 1
                        bbox_text = None if last_person is None else last_person["bbox_xyxy"]
                        print(
                            f"\nTrigger {mode} for {state.workstation_id}, "
                            f"frames={frame_count}, presence={state.stable_presence}, bbox={bbox_text}"
                        )
                except Exception as exc:
                    print(f"Failed to prepare VL input: {exc}")

            if (
                report_worker is not None
                and now - last_report_time >= args.report_interval
            ):
                report = state.build_report(args.report_interval)
                if report_worker.trigger(report):
                    print("\nTrigger workstation VL report...")
                    last_report_time = now

            status = (
                f"frame={frame_id} yolo={last_yolo_ms:.1f}ms "
                f"presence={state.stable_presence} activity={state.stable_activity}"
            )
            if vl_worker is not None:
                running, error, elapsed_s = vl_worker.get_status()
                is_agent = isinstance(vl_worker, AgentActivityWorker)
                label = "Agent" if is_agent else "VL"
                if running:
                    status += f" | {label} reasoning..."
                elif error:
                    status += f" | {label} error"
                elif elapsed_s > 0:
                    status += f" | {label}={elapsed_s:.1f}s"
            if report_worker is not None:
                _, report_elapsed, report_error, report_running = report_worker.get_status()
                if report_running:
                    status += " | report..."
                elif report_error:
                    status += " | report error"
                elif report_elapsed > 0:
                    status += f" | report={report_elapsed:.1f}s"

            display = draw_overlay(frame.copy(), state, last_person, status)
            if not args.no_window:
                cv2.imshow("WorkGuard-VL", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if cap is not None and frame_delay_s > 0:
                sleep_s = frame_delay_s - (time.perf_counter() - loop_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as exc:
        print(f"Runtime failed: {exc}")
    finally:
        print("\nFinal structured report:")
        print(json.dumps(state.build_report(None), ensure_ascii=False, indent=2))
        print("\nPerformance stats:")
        print(format_latency_stats("YOLO latency", yolo_latency_ms, "ms"))
        if vl_worker is not None:
            print(format_latency_stats("VL analysis time", vl_worker.elapsed_history_s, "s"))
        if report_worker is not None and not args.disable_report:
            report = state.build_report(None)
            print("\nDeterministic activity counts:")
            print(json.dumps(report.get("activity_counts", {}), ensure_ascii=False, indent=2))
        if zed is not None:
            zed.close()
        if cap is not None:
            cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
