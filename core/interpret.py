from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from pathlib import Path

from schemas.core import Event, Instruction, Program, State

RUN_TIMEOUT_SECONDS = 30


def _run_command_safe(command: str, cwd: str, timeout: float = RUN_TIMEOUT_SECONDS) -> dict[str, object]:
    """
    Execute a RUN command safely by splitting into argv, disabling shell execution,
    and enforcing a timeout. Returns stdout/stderr/exit_code with timing metadata.
    """

    if not isinstance(command, str) or not command.strip():
        raise ValueError("RUN instruction requires non-empty command")

    cmd_parts = shlex.split(command)
    if not cmd_parts:
        raise ValueError("RUN instruction requires non-empty command")

    started_at = time.time()
    try:
        proc = subprocess.run(
            cmd_parts,
            shell=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        ended_at = time.time()
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "started_at": started_at,
            "ended_at": ended_at,
        }
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - covered via behavior check
        ended_at = time.time()
        return {
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\n[timeout expired]",
            "exit_code": -1,
            "started_at": started_at,
            "ended_at": ended_at,
        }


def _apply_edit(payload: dict[str, object], repo_root: str) -> dict[str, object]:
    file_path = payload.get("file_path")
    if not isinstance(file_path, str):
        raise ValueError("EDIT instruction requires 'file_path' string in payload")
    content = payload.get("content", "")

    repo_root_path = Path(repo_root).resolve()
    target_path = (repo_root_path / file_path).resolve()
    try:
        target_path.relative_to(repo_root_path)
    except ValueError as exc:  # pragma: no cover - exercised in security test
        raise ValueError("EDIT path escapes repo_root") from exc

    target_path.parent.mkdir(parents=True, exist_ok=True)
    content_str = str(content)
    target_path.write_text(content_str, encoding="utf-8")
    return {"file_path": file_path, "bytes_written": len(content_str)}


def interpret(state: State, program: Program, job_id: str, step_id: int) -> tuple[State, list[Event]]:
    events: list[Event] = []
    current_step = step_id
    for instruction in program.instructions:
        if instruction.kind == "RUN":
            cmd = instruction.payload.get("cmd")
            if not isinstance(cmd, str):
                raise ValueError("RUN instruction requires 'cmd' string in payload")
            result = _run_command_safe(cmd, state.repo_root)
            event = Event(
                event_id=str(uuid.uuid4()),
                job_id=job_id,
                step_id=current_step,
                type="RUN",
                payload={
                    "cmd": cmd,
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "exit_code": result["exit_code"],
                },
                started_at=result["started_at"],
                ended_at=result["ended_at"],
            )
            if result["exit_code"] != 0:
                state.diagnostics.last_error = (
                    f"RUN failed (exit_code={result['exit_code']}): {cmd}"
                )
            events.append(event)
        elif instruction.kind == "EDIT":
            try:
                edit_result = _apply_edit(instruction.payload, state.repo_root)
                event_payload = edit_result
            except Exception as exc:  # noqa: BLE001 - broad for error payload capture
                state.diagnostics.last_error = str(exc)
                event_payload = {
                    "file_path": instruction.payload.get("file_path"),
                    "error": str(exc),
                }
            event = Event(
                event_id=str(uuid.uuid4()),
                job_id=job_id,
                step_id=current_step,
                type="EDIT",
                payload=event_payload,
                started_at=time.time(),
                ended_at=time.time(),
            )
            events.append(event)
        elif instruction.kind == "META":
            payload = instruction.payload
            allowed_modes = {"OK", "DEGRADED_PARTIAL", "DOWN"}
            memory_mode = payload.get("memory_mode")
            if isinstance(memory_mode, str) and memory_mode in allowed_modes:
                state.diagnostics.memory_mode = memory_mode
            last_error = payload.get("last_error")
            if isinstance(last_error, str):
                state.diagnostics.last_error = last_error
            event = Event(
                event_id=str(uuid.uuid4()),
                job_id=job_id,
                step_id=current_step,
                type="META",
                payload=payload,
                started_at=time.time(),
                ended_at=time.time(),
            )
            events.append(event)
        current_step += 1

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=state.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            state.git_head = out.stdout.strip()
    except Exception:
        pass

    return state, events
