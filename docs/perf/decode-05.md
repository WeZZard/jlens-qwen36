# decode-05 — Free the event loop: GPU work on worker threads

**Code:** `jlens_qwen/serve.py` (`_gpu_lock`, `_sample_tok`,
`asyncio.to_thread` around the three GPU call sites in `chat_stream`).

## The problem

Every MLX call in the chat stream — prefill `extend`, per-token `extend`,
the readout — ran on uvicorn's event-loop thread. During generation the
loop was blocked more than it was free: a trivial `GET /api/model` took
**342 ms median (max 590 ms)**, and `/api/chat_control` — the pause
button — took **340 ms** to even respond, worst during multi-second
prefills.

## The fix

MLX releases the GIL during compute — measured directly: an asyncio
ticker saw a **1063 ms** worst tick with 10 decode steps inline vs
**1.1 ms** with the same work in `asyncio.to_thread`. So the fix is
minimal: wrap the three blocking call sites in `to_thread`, under a
module-level `asyncio.Lock` that serializes GPU access across concurrent
streams (preserving the previous implicit serialization; MLX
thread-safety across concurrent evals is otherwise unproven here).
Sampling moved into the same locked section (`_sample_tok`).

## Result (end-to-end on the real server, during generation)

| metric | before | after |
|---|---|---|
| `GET /api/model` median | 341.9 ms | **0.8 ms** |
| `GET /api/model` max | 589.9 ms | **3.3 ms** |
| pause round-trip | 339.5 ms | **1.4 ms** |
| probe throughput (3.5 s window) | 9 requests | 145 requests |

Decode throughput unchanged (117.6 ms/token, within the 114.5–118.1
variance band — the change is orchestration only).

## Verification

Gate 4/4 (`tests/test_decode_gate.py`) — decode math untouched. The
responsiveness probe exercised the full stream against the live server:
prefill + 40 generated snapshots, a pause + resume round-trip
mid-generation, and a clean `done` event.
