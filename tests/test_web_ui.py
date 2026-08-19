from pathlib import Path


def test_web_ui_exposes_v0_controls_without_seed():
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert 'id="adapter"' in html
    assert 'id="bpm"' in html
    assert 'id="bars"' in html
    assert 'id="temperature"' in html
    assert 'id="generate"' in html
    assert 'id="download"' in html
    assert 'id="seed"' not in html
    assert "onnxruntime-web" in html
    assert "./app.mjs" in html


def test_pages_workflow_builds_registry_and_adapter_assets():
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "scripts/build_registry.py" in workflow
    assert "--web-root web" in workflow


def test_browser_runtime_config_defaults_to_wasm_and_requires_model_asset():
    config = Path("web/runtime-config.json").read_text(encoding="utf-8")
    assert '"model_url": ""' in config
    assert '"wasm"' in config
