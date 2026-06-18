# WorkGuard-VL Agent Interview Notes

## One-Minute Pitch

WorkGuard-VL Agent is a local multimodal monitoring Agent for a single workstation. YOLO handles real-time person detection and ROI tracking, while Qwen3.5-VL analyzes recent frame windows to classify behaviors such as working, discussion, phone use, sleeping, short leave, long leave, and off-duty. The Agent can autonomously call tools for employee lookup, schedule lookup, policy RAG, history query, and time checks. Alerts are not directly controlled by the LLM; they pass through deterministic guardrails such as confidence thresholds, frame-count checks, cooldown, and fallback review.

## Architecture

```text
Video / ZED2i
  -> YOLO person tracking
  -> ROI filter
  -> Workstation state machine
  -> Frame buffer
  -> Qwen3.5-VL ReAct Agent
  -> Tools / RAG / Memory
  -> Rule-guarded alerting
  -> Structured report
```

## Key Technical Points

### 1. ROI-Based Person Filtering

YOLO may detect many people in the scene. The system filters detections by checking whether the person bounding-box center falls inside the workstation ROI. This keeps the Agent focused on the target workstation instead of nearby people.

### 2. Presence State Machine

The workstation state machine separates detection noise from semantic state:

- `occupied`: person detected inside ROI
- `recently_left`: person disappeared recently
- `vacant`: no person for longer than the short-leave threshold

This prevents a single missed detection from immediately becoming a long-leave event.

### 3. Short Temporal Window

Single-frame classification is brittle. The Agent receives sampled frames from the last few seconds, for example 5 frames over 12 seconds. This helps distinguish:

- short drinking vs primary work
- brief posture changes vs sleeping
- momentary phone glance vs sustained phone use
- short leave vs empty workstation

### 4. Vacant-State Reasoning

The Agent continues to run when no person is in ROI. It receives:

- current empty ROI image
- absence duration
- long-leave threshold
- current time
- schedule/tool access

This allows it to output `短暂离岗`, `长时间离岗`, or `下班`.

### 5. ReAct Tool Calling

The Agent uses text-based ReAct instead of relying on vendor-specific function calling:

```text
思考: I need schedule information.
动作: ACTION: query_schedule(employee_id="E001")
```

The Python loop parses `ACTION`, executes the tool, stores the result in short-term memory, and calls the model again.

### 6. RAG Policy Retrieval

Policy documents are loaded from Markdown/TXT/JSON, split into chunks, embedded with SentenceTransformer, indexed by FAISS, and retrieved by semantic similarity. The Agent can call `search_policies` when it needs evidence for compliance decisions.

### 7. Memory

Short-term memory stores the current reasoning chain: observations, thoughts, actions, and tool results. Long-term memory stores behavior distributions per workstation. This gives the Agent temporal context without overloading the prompt.

### 8. LLM Guardrails

The LLM can propose `alerts_sent=true`, but Python decides whether an alert is allowed. Guardrails include:

- violation activity whitelist
- confidence threshold
- minimum frame count for phone-use alerts
- cooldown for repeated alerts
- no auto-alert on fallback results
- no direct `send_alert` tool calls from the LLM

This is important because production systems should not let a generative model directly execute high-risk actions.

### 9. Failure Handling

Observed issues and fixes:

- Repeated tool calls: detect duplicate tool+args and stop with fallback.
- Multiple ACTIONs in one response: stop and fallback.
- Long model outputs: reduce per-step generation tokens.
- Weak report response such as “收到”: use deterministic summary fallback.
- Single-frame false phone alert: require minimum frame count and confidence.

## Good Interview Answers

### Why YOLO + VL instead of only VL?

YOLO is fast and stable for localization. VL models are slower but better at semantic interpretation. The split reduces latency and lets the VL model focus on a cropped/annotated workstation context.

### Why use multi-frame input?

Office behavior is temporal. A single frame can misclassify drinking, turning, resting, or looking down. A short frame window captures consistency and reduces false positives.

### Why keep alerting outside the LLM?

Alerts are side effects. The LLM can reason, but deterministic code should enforce thresholds, cooldown, and compliance rules. This makes the system safer and easier to debug.

### What are the main limitations?

- Small local VL models can hallucinate details.
- ROI calibration matters.
- Complex social behaviors are ambiguous.
- True production use needs privacy review, audit logging, access control, and human-in-the-loop escalation.

## Suggested Demo Flow

1. Show ROI config and YOLO tracking.
2. Show normal work classification.
3. Show discussion/drinking as non-violations.
4. Show phone-use detection with multi-frame evidence.
5. Show empty ROI becoming short leave, then long leave/off-duty depending on schedule.
6. Open logs and show ReAct steps and tool results.

