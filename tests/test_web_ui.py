from pathlib import Path


def test_web_ui_exposes_base_adapter_controls_without_seed():
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="base"' in html
    assert 'id="adapter"' in html
    assert 'id="bpm"' in html
    assert 'id="bars"' in html
    assert 'id="temperature"' in html
    assert 'id="generate"' in html
    assert 'id="download"' in html
    assert 'id="seed"' not in html
    assert "onnxruntime-web" in html
    assert "./app.mjs" in html


def test_pages_workflow_builds_base_and_adapter_assets():
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "scripts/build_registry.py" in workflow
    assert "--bases bases" in workflow
    assert "--adapters adapters" in workflow
    assert "--web-root web" in workflow


def test_browser_app_reads_both_registries():
    app = Path("web/app.mjs").read_text(encoding="utf-8")
    assert "./data/bases.json" in app
    assert "./data/adapters.json" in app
    assert "adapter.base_model" in app
    assert "base.checkpoint_sha256" in app
