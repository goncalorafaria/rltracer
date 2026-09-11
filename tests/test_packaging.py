from __future__ import annotations

import importlib.resources
from pathlib import Path

import rltracer


def test_base_package_has_no_jtc_dependency() -> None:
    source_root = Path(__file__).parents[1] / "src" / "rltracer"
    for path in source_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import jtc_data_commons" not in source
        assert "from jtc_data_commons" not in source


def test_web_explorer_assets_are_packaged() -> None:
    root = importlib.resources.files("rltracer.viz")
    assert root.joinpath("index.html").is_file()
    assert root.joinpath("app.js").is_file()


def test_public_api_comes_from_standalone_package() -> None:
    assert rltracer.RLTracer.__module__ == "rltracer.tracer"
