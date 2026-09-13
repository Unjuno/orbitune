from pathlib import Path


def test_web_root_exposes_latest_a2_compound_controls():
    html = Path("web/index.html").read_text(encoding="utf-8")
    for dom_id in (
        "compound-variant",
        "compound-model-meta",
        "compound-temperature",
        "compound-top-p",
        "compound-live-start",
        "compound-live-pause",
        "compound-live-stop",
        "compound-offline",
        "compound-install",
        "compound-generate",
        "compound-download",
        "compound-status",
    ):
        assert f'id="{dom_id}"' in html
    assert "A2-512" in html
    assert "onnxruntime-web@1.29.0" in html
    assert "./compound-app.mjs" in html
    assert "./app.mjs" not in html
    assert "Theory-REMI" not in html


def test_pages_workflow_builds_base_and_adapter_assets():
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "scripts/build_registry.py" in workflow
    assert "--bases bases" in workflow
    assert "--adapters adapters" in workflow
    assert "--web-root web" in workflow


def test_legacy_browser_module_still_reads_both_registries():
    app = Path("web/app.mjs").read_text(encoding="utf-8")
    assert "./data/bases.json" in app
    assert "./data/adapters.json" in app
    assert "adapter.base_model" in app
    assert "base.checkpoint_sha256" in app
