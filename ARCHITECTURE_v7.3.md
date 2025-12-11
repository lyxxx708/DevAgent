# DevAgent v7.3 Architecture Summary

## Layered Responsibilities (L0–L4)
- **L0 – Reality**: The concrete execution environment: git repo, filesystem, and CI/test runtime where artifacts live.
- **L1 – S-pipe (Semantic Kernel)**: Runs `interpret(State, Program) -> (State', Events)` and emits events for downstream use.
- **L2 – Multi-dimensional Memory + Views + Store**: Maintains memory, store, and view materialization that contextualize S/F/T pipelines.
- **L3 – DevAgent (Main LLM)**: Primary agent loop that consumes views and produces actions within the defined pipelines.
- **L4 – Meta (Planner + Controller)**: LLMMetaPlanner and MetaController that steer the agent by selecting programs and controls.

## Pipelines
- **S-pipe (State Pipeline)**: Semantic execution that interprets inputs and produces new state plus events.
- **F-pipe (Focus Pipeline)**: Decides which files/modules/tests and memory slices to attend to, assembling focus views.
- **T-pipe (Trace Pipeline)**: Captures decision traces and audit data for observability and replay.

## Key Modules
- **config/settings**: Centralized configuration and environment toggles that other modules consume.
- **schemas**: Contracts for data structures
  - *core*: Definitions for interpretation inputs/outputs and agent actions.
  - *memory*: Schemas for memory records, vector payloads, and retrieval signals.
  - *views*: Formats for focus views and rendered outputs.
  - *meta*: Structures for meta-plans, controller directives, and planner exchanges.
- **infra**
  - *observer*: UnifiedObserver implementation for traces/events across S/F/T pipelines.
  - *vector_store*: Persistence and similarity backend supporting memory operations.
- **store**
  - *event_store*: Durable ledger of events entering S-pipe.
  - *trace_ledger*: Structured trace storage for reasoning steps and observations.
- **memory**
  - *store*: High-level memory API bridging schemas, vector store, and trace/event stores.
  - *ingest*: Pipelines that convert observations into stored memory entries.
  - *selector*: Retrieval and filtering logic to fetch relevant memories.
  - *reranker*: Reranking to refine selector outputs for F-pipe consumption.
- **core/interpret**: Interpretation engine that turns focused context into actions or responses and feeds T-pipe reasoning.
- **views/focus**: View builders that assemble focus contexts and user-facing slices from selectors/rerankers.
- **agent/devagent**: DevAgent orchestration tying pipelines, memory, views, and interpretation into runnable behaviors.
- **meta**
  - *llm_meta_planner*: Higher-level planner driving T-pipe steps and decompositions.
  - *controller*: Execution controller that applies planner directives to the agent runtime.
- **task/runner**: Executes tasks, manages lifecycle, and interfaces with agent outputs.
- **api/http**: HTTP surface exposing agent capabilities and status.

## DevAgentMode
- **OPTIMIZED_STRUCTURED**: Primary structured execution using the DevAgent and pipelines.
- **BOOTSTRAP_LLM_HEAVY**: Bootstrap-friendly flow that leans on the LLM while maintaining the same pipeline semantics.

(Plan-only style APIs may layer on top later but are subordinate to the official DevAgentMode values above.)
