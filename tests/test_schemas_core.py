from schemas.core import Event, Instruction, Program, State, StateDiagnostics


def test_state_and_program_serialization():
    diagnostics = StateDiagnostics()
    state = State(git_head="abc123", repo_root="/repo", diagnostics=diagnostics)
    instruction = Instruction(kind="RUN", payload={"cmd": "echo"})
    program = Program(instructions=[instruction])

    assert state.diagnostics.memory_mode == "OK"
    assert program.instructions[0].kind == "RUN"

    event = Event(
        event_id="e1",
        job_id="j1",
        step_id=1,
        type="RUN",
        payload={"status": "ok"},
        started_at=0.0,
        ended_at=1.0,
    )
    dumped = event.model_dump()
    assert dumped["job_id"] == "j1"
    assert dumped["type"] == "RUN"
