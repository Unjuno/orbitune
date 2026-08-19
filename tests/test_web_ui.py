from pathlib import Path


def test_web_ui_exposes_v0_controls_without_seed():
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="adapter"' in html
    assert 'id="bpm"' in html
    assert 'id="bars"' in html
    assert 'id="temperature"' in html
    assert 'id="generate"' in html
    assert 'id="seed"' not in html
    assert "./data/adapters.json" in html


def test_pages_workflow_bundles_registry():
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "cp registry/adapters.json web/data/adapters.json" in workflow
