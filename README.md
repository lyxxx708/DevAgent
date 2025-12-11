# DevAgent v7.3

DevAgent v7.3 is a layered agent architecture that keeps execution deterministic and offline-friendly while leaving room for future LLM integration. It follows L0–L4 layers and S/F/T pipelines:

- **L0 – Reality:** Git repo, filesystem, CI/test runtime.
- **L1 – S-pipe (Semantic Kernel):** `core.interpret.interpret(State, Program) -> (State', Events)` executes RUN/EDIT/META instructions.
- **L2 – Memory + Views + Store:** structured stores (EventStore, TraceLedger, MemoryStore, optional VectorStore) plus selector/reranker and focus builders.
- **L3 – DevAgent:** orchestrates interpret → observer → memory ingest/selection/rerank → focus view → `DecisionInputView`.
- **L4 – Meta:** `LLMMetaPlanner` (heuristic, non-LLM) + `MetaController` steer DevAgent and emit trace entries. `TaskRunner` manages per-job lifecycle, and the HTTP API wraps TaskRunner.

## Components and flow
- **DevAgent (agent/devagent.py):** Runs a single step, persists events via `UnifiedObserver`, ingests memory, and produces decision context.
- **MetaController (meta/controller.py):** Builds `MetaInputView`, invokes `LLMMetaPlanner` to get a `MetaPlan`, delegates to DevAgent, and records a `TraceEntry`.
- **TaskRunner (task/runner.py):** Holds per-job `State`, `GoalView`, and step counters; delegates each step to MetaController.
- **HTTP API (api/http.py):** FastAPI surface exposing `/health`, `/jobs`, `/jobs/{job_id}/steps` on top of TaskRunner.
- **Stores/infra:** EventStore, TraceLedger, MemoryStore, optional VectorStore; UnifiedObserver bridges events/traces into persistence.

## Dependencies and optional features
Core logic is pure Python. Some capabilities rely on optional packages: `pydantic`, `sqlmodel`, `fastapi`, `uvicorn`, `faiss`, and `numpy`. In minimal/offline setups, only a subset of tests may run; full functionality requires installing these dependencies (see `requirements.txt`).

### Trust model & safeguards
- Intended for trusted/offline use; avoid exposing the HTTP API or accepting arbitrary commands/goals from untrusted clients without adding auth/ACLs or stricter allowlists.
- RUN commands execute without `shell=True` and enforce a timeout to limit command injection surface.
- EDIT instructions are confined to the configured `repo_root`; paths escaping this root are rejected.

## Getting started
- Quick sanity check (no external services):
  ```bash
  python -m compileall .
  ```
- Full test suite (requires optional deps installed):
  ```bash
  pytest
  ```
- Start the HTTP server (after installing dependencies):
  ```bash
  uvicorn api.http:app --reload
  ```

## Minimal HTTP flow
1. Create a job via `POST /jobs` with `repo_root` and a `GoalView` payload.
2. Execute a step via `POST /jobs/{job_id}/steps` with a `Program` (e.g., a failing RUN command) and optional `AgentHints`.
3. Inspect the returned `DecisionInputView` for `focus_view.files`, `memory_view.items`, `mode`, and `goal_view` to guide the next action.

See `docs/QUICKSTART_DEVAGENT_V7_3.md` for a step-by-step walkthrough, and `ARCHITECTURE_v7.3.md` for full design details. The included `LLMMetaPlanner` is a heuristic, non-LLM stub that can be swapped for a real LLM planner without changing public APIs.

## Programmatic usage (without HTTP)
Run the same DevAgent/MetaController/TaskRunner stack locally without FastAPI for quick experiments:

```bash
python examples/run_fix_failures_local.py /path/to/repo
```

This script wires the full v7.3 pipeline in-process, executes a failing RUN instruction to exercise focus/memory pipelines, and prints a concise summary. It mirrors the HTTP stack but keeps everything local; see `docs/QUICKSTART_DEVAGENT_V7_3.md` for the HTTP flow.
