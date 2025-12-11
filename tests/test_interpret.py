import tempfile

from schemas.core import Event, Instruction, Program, State, StateDiagnostics
from core.interpret import interpret


def test_interpret_run_sets_diagnostics_with_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        state = State(git_head="head", repo_root=tmpdir)
        program = Program(
            instructions=[Instruction(kind="RUN", payload={"cmd": "python -c \"import sys; sys.exit(1)\""})]
        )
        updated_state, events = interpret(state, program, job_id="job", step_id=1)
        assert events[0].type == "RUN"
        assert "exit_code=1" in updated_state.diagnostics.last_error
        assert "python -c" in updated_state.diagnostics.last_error


def test_interpret_meta_applies_typed_updates_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        state = State(git_head="head", repo_root=tmpdir, diagnostics=StateDiagnostics())
        payload_valid = {"memory_mode": "DEGRADED_PARTIAL", "last_error": "oops", "ignored": 123}
        payload_invalid = {"memory_mode": "INVALID", "last_error": None}
        program = Program(
            instructions=[
                Instruction(kind="META", payload=payload_valid),
                Instruction(kind="META", payload=payload_invalid),
            ]
        )
        updated_state, events = interpret(state, program, job_id="job", step_id=1)
        assert updated_state.diagnostics.memory_mode == "DEGRADED_PARTIAL"
        assert updated_state.diagnostics.last_error == "oops"
        assert events[0].payload == payload_valid
        assert events[1].payload == payload_invalid
