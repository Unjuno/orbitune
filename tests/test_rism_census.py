from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "rism_census.py"
SPEC = importlib.util.spec_from_file_location("rism_census", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_person_date_pd_safety_is_fail_closed() -> None:
    safe = MOD._person_date_is_pd_safe
    assert safe("1685-1750")
    assert safe("fl. 1732-1735")
    assert safe("-12.12.1803")
    assert safe("1870-1955")
    assert not safe("1870-1956")
    assert not safe("1900-")
    assert not safe("")
    assert not safe("unknown")


def test_incipit_fingerprint_is_stable_under_whitespace() -> None:
    base = {"clef": "G-2", "keysig": "xF", "timesig": "3/4", "pae": "4C D E /"}
    spaced = {"clef": " G-2 ", "keysig": "xF", "timesig": "3/4", "pae": "4C   D E /"}
    assert MOD._incipit_fingerprint(base) == MOD._incipit_fingerprint(spaced)


def test_reservoir_never_exceeds_limit() -> None:
    import random

    reservoir: list[dict[str, str]] = []
    rng = random.Random(1)
    for seen in range(1, 101):
        MOD._reservoir_add(reservoir, {"pae": str(seen)}, seen=seen, limit=7, rng=rng)
    assert len(reservoir) == 7


def test_verovio_logging_uses_logging_api_not_toolkit_option() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"logLevel"' not in source
    assert "enableLogToBuffer" in source
    assert "enableLog(verovio.LOG_WARNING)" in source
