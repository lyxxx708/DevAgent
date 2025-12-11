# DevAgent v7.3 Quickstart

This quickstart introduces the main building blocks of DevAgent v7.3 and shows how to exercise the HTTP API locally.

## Architecture at a Glance
- **DevAgent (L3)** orchestrates the S/F/T pipelines: runs the core interpreter, ingests events into memory, builds focus/memory/state views, and produces a `DecisionInputView` for downstream planners.
- **MetaController (L4)** constructs a `MetaInputView`, calls the heuristic `LLMMetaPlanner`, delegates execution to DevAgent, and records decision traces.
- **TaskRunner** manages per-job lifecycle (state, goal, step counters) and delegates execution to the MetaController.
- **HTTP API** (`api/http.py`) exposes a thin FastAPI surface over TaskRunner.
- **Stores & Infra**
  - `EventStore` / `TraceLedger` persist events and decision traces.
  - `MemoryStore` (with optional `VectorStore`) holds structured memory items used by selectors/rerankers.
  - `UnifiedObserver` forwards emitted events and traces to the stores.

## Running the HTTP Server
Ensure dependencies from `requirements.txt` are installed, then start FastAPI via uvicorn:

```bash
uvicorn api.http:app --reload
```

The server boots a single DevAgent stack using paths from your `.env`/`Settings`.

**Trust & safety:** This stack is intended for trusted/local use. RUN commands execute without `shell=True` and include a timeout, and EDIT paths are confined to the supplied `repo_root`. Avoid exposing the API to untrusted networks without additional auth/ACLs or stricter command policies.

## Minimal API Flow
1. **Create a job**
   ```bash
   curl -X POST http://localhost:8000/jobs \
     -H "Content-Type: application/json" \
     -d '{
       "repo_root": "/path/to/repo",
       "goal": {"task_type": "fix_failures", "natural_language_goal": "Make tests pass"}
     }'
   ```
   Response includes `job_id`.

2. **Run a step**
   ```bash
   curl -X POST http://localhost:8000/jobs/<job_id>/steps \
     -H "Content-Type: application/json" \
     -d '{
       "program": {"instructions": [{"kind": "RUN", "payload": {"cmd": "pytest"}}]},
       "hints": null
     }'
   ```
   The response `decision` contains:
   - `focus_view.files` — files/tests the agent should attend to.
   - `memory_view.items` — structured memories derived from previous events.
   - `goal_view` and `mode` — confirm the job goal and execution mode.

3. **Inspect persisted state (optional)**
   - `EventStore` and `TraceLedger` track events and decision traces under the configured database paths.
   - `MemoryStore` accumulates `MemoryItem` entries from RUN/EDIT events.

This flow exercises DevAgent v7.3 end-to-end without any LLM or network dependencies.
