# WorkGuard-VL Agent

WorkGuard-VL Agent is a single-workstation visual monitoring Agent for office/lab scenarios. It combines YOLO person tracking, short-window multimodal reasoning, ReAct-style tool calls, RAG policy retrieval, schedule/employee lookup, and rule-guarded alerting.

The project is designed as a practical local demo: YOLO handles fast perception, Qwen3.5-VL via Ollama handles behavior reasoning, and Python rules keep high-risk actions such as alerts under deterministic control.

## Features

- Single-workstation ROI monitoring with YOLO person tracking.
- Short temporal window input: sends recent sampled frames instead of only one frame.
- Occupancy states: `occupied`, `recently_left`, `vacant`.
- Behavior labels: `工作`, `聊天/讨论`, `玩手机`, `睡觉`, `喝水`, `短暂离岗`, `长时间离岗`, `下班`, `离开`, `其他`.
- ReAct Agent mode with autonomous tool calls:
  - `query_employee`
  - `query_schedule`
  - `search_policies`
  - `query_activity_history`
  - `get_current_time`
- RAG policy search with SentenceTransformer + FAISS.
- Guardrails for alerting:
  - alert cooldown
  - minimum frame count for phone-use alerts
  - no direct `send_alert` tool call by the LLM
  - fallback results require review and do not auto-alert
- Vacant workstation reasoning: keeps analyzing after a person leaves ROI and distinguishes short leave, long leave, and off-duty.

## Project Structure

```text
workguard-vl-agent/
  workstation_vl_agent.py      # Main runtime: video/camera, YOLO, state machine, workers
  llm_client.py                # Ollama multimodal calls and report generation
  seats_config.example.json    # Example workstation ROI config
  agent/
    react.py                   # ReAct loop, action parser, guardrails
    tools.py                   # Tool registry and local tools
    rag.py                     # Markdown/TXT/JSON loader, chunking, FAISS retrieval
    memory.py                  # Short-term and long-term Agent memory
  data/
    employees.json             # Example employee data
    schedule.json              # Example schedule/activity log
    policies.md                # Example policy knowledge base
  docs/
```

## Requirements

- Python 3.10+
- Ollama running locally
- A multimodal Ollama model, tested with `qwen3.5:9b`
- YOLO model file, such as `yolo11n.engine`, `yolo11n.pt`, or another Ultralytics-compatible model
- Optional: ZED SDK + `pyzed` for live ZED2i camera mode

Install Python dependencies:

```bat
pip install -r requirements.txt
```

Pull the model:

```bat
ollama pull qwen3.5:9b
```

## Run

Offline video example:

```bat
python workstation_vl_agent.py ^
  --agent ^
  --agent-mode react ^
  --vl-model qwen3.5:9b ^
  --model D:\path\to\yolo11n.engine ^
  --video D:\path\to\video.mp4 ^
  --workstations-config seats_config.example.json
```

Recommended runtime parameters:

```bat
python workstation_vl_agent.py ^
  --agent ^
  --agent-mode react ^
  --vl-model qwen3.5:9b ^
  --vl-interval 30 ^
  --vl-window-s 12 ^
  --vl-sample-every-s 3 ^
  --vl-max-frames 5 ^
  --vacant-vl-interval 30 ^
  --long-left-s 300 ^
  --alert-cooldown-s 300
```

Use `--agent-mode prefetch` for a faster, more stable demo path. Use `--agent-mode react` to demonstrate autonomous tool calling.

## How It Works

1. Video frames are read from an offline video or ZED2i camera.
2. YOLO tracks people and filters detections by workstation ROI.
3. A state machine updates workstation presence:
   - person in ROI -> `occupied`
   - recently left -> `recently_left`
   - no person beyond threshold -> `vacant`
4. The Agent receives a short temporal window of frames while occupied.
5. If no person is detected, the Agent still receives the current ROI frame and absence duration.
6. The ReAct loop can call local tools for employee, schedule, history, policy, and time information.
7. Python guardrails decide whether an alert can actually be sent.
8. Structured reports and performance stats are printed at shutdown.

## Notes

- Do not commit large videos, model weights, TensorRT engines, generated logs, or FAISS cache files.
- `data/*.json` and `data/policies.md` are demo data. Replace them with your own non-sensitive data before public release.
- This project is a research/demo prototype, not a production HR or compliance system.
