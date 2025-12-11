from __future__ import annotations

from pathlib import Path

import pytest

from core.interpret import interpret
from schemas.core import Instruction, Program, State


def test_run_uses_shell_false_and_timeout(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_run(cmd, shell, cwd, capture_output, text, check, timeout):  # type: ignore[override]
        captured.update({
            "cmd": cmd,
            "shell": shell,
            "timeout": timeout,
            "cwd": cwd,
            "capture_output": capture_output,
            "text": text,
            "check": check,
        })

        class Result:
            stdout = ""
            stderr = ""
            returncode = 0

        return Result()

    monkeypatch.patch("core.interpret.subprocess.run", fake_run)

    state = State(git_head="", repo_root=str(tmp_path))
    program = Program(instructions=[Instruction(kind="RUN", payload={"cmd": "echo hi"})])

    interpret(state, program, job_id="job", step_id=1)

    assert isinstance(captured.get("cmd"), list)
    assert captured.get("shell") is False
    assert captured.get("timeout") is not None and captured["timeout"] > 0


def test_edit_rejects_path_traversal(tmp_path):
    state = State(git_head="", repo_root=str(tmp_path))
    program = Program(
        instructions=[Instruction(kind="EDIT", payload={"file_path": "../outside.txt", "content": "data"})]
    )

    new_state, events = interpret(state, program, job_id="job", step_id=1)

    outside_path = (Path(tmp_path) / ".." / "outside.txt").resolve()
    assert not outside_path.exists()

    assert events
    edit_event = events[0]
    assert edit_event.type == "EDIT"
    assert isinstance(edit_event.payload, dict)
    assert "error" in edit_event.payload
    assert new_state.diagnostics.last_error
