from orbitune.evaluation import evaluate_events
from orbitune.events import NoteEvent


def test_evaluate_events_reports_basic_structure():
    events = [
        NoteEvent(bar=0, position=0, pitch=60, duration=4, velocity=64),
        NoteEvent(bar=0, position=4, pitch=64, duration=4, velocity=72),
        NoteEvent(bar=1, position=0, pitch=67, duration=8, velocity=80),
    ]
    report = evaluate_events(events)
    assert report["valid"] is True
    assert report["notes"] == 3
    assert report["bars"] == 2
    assert report["pitch_min"] == 60
    assert report["pitch_max"] == 67
    assert report["notes_per_bar_max"] == 2
