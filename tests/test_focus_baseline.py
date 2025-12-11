from schemas.core import Event
from views.focus import BaselineFocusInferer


def test_baseline_focus_inferer_extracts_paths():
    inferer = BaselineFocusInferer()
    event = Event(
        event_id="e1",
        job_id="job",
        step_id=1,
        type="RUN",
        payload={
            "exit_code": 1,
            "stderr": "E   AssertionError: fail in src/module.py\nFailed test in tests/test_module.py:12",
        },
        started_at=0.0,
        ended_at=0.1,
    )
    success_event = Event(
        event_id="e2",
        job_id="job",
        step_id=2,
        type="RUN",
        payload={"stderr": "ignored without exit code"},
        started_at=0.2,
        ended_at=0.3,
    )
    focus = inferer.infer([event, success_event])
    assert "src/module.py" in focus.files
    assert "tests/test_module.py" in focus.tests
    assert len(focus.files) == 1
    assert len(focus.tests) == 1
