# Log Review: 2026-06-18 Run

Source log reviewed: `workguard_20260618_201929.log`

## Summary

The Agent pipeline ran end-to-end:

- YOLO detection and tracking were stable.
- Temporal-window input worked: most Agent calls used 4 frames.
- Vacant-state Agent reasoning worked: `recently_left` and `vacant` states continued to trigger Agent calls.
- The Agent produced `短暂离岗` while absence duration was below the long-leave threshold.
- Final activity counts showed:
  - `工作`: 20
  - `聊天/讨论`: 2
  - `玩手机`: 5
  - `短暂离岗`: 5

Performance:

- YOLO latency: mean about 16.62 ms, p95 about 28.05 ms.
- Agent/VL analysis: mean about 12.27 s, p50 about 11.19 s, p95 about 23.65 s.

## Issues Found

### 1. Single-Frame Phone Alert Risk

The first Agent call used only one frame and still produced a phone-use alert. This is risky because phone use should require short-window evidence.

Fix:

- Added `--phone-alert-min-frames`.
- Phone-use alerts now require enough temporal-window frames.

### 2. Repeated Alerts During Same Event

Phone-use behavior produced multiple alerts across consecutive Agent calls.

Fix:

- Added `--alert-cooldown-s`.
- Repeated alerts for the same workstation/activity are suppressed during cooldown.

### 3. Fallback Result Auto-Alert

Some fallback results such as "Agent 工具调用被拦截" still triggered alerts.

Fix:

- Fallback results now set `alerts_sent=false`.
- Fallback details explicitly say human review is required.

### 4. Duration Hallucination

The Agent sometimes described a 12-second frame window as "30 seconds" or "20 minutes".

Fix:

- Added prompt instruction: do not claim durations beyond the provided temporal window.

### 5. Vacant-State Coverage

Earlier versions did not call the Agent after the person left ROI.

Current behavior:

- `recently_left` and `vacant` now trigger periodic Agent calls.
- Absence duration and long-leave threshold are injected into the prompt.
- The Agent can classify `短暂离岗`, `长时间离岗`, or `下班`.

## Current Status

The system is suitable for a GitHub demo project after the guardrail fixes:

- Visual pipeline works.
- ReAct tool calling works.
- Vacant-state reasoning works.
- Alerting has deterministic safety checks.
- Remaining limitations are mainly model quality, ROI calibration, and privacy/production hardening.

