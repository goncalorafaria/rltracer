from __future__ import annotations

from pathlib import Path

import rltracer


def test_base_package_has_no_jtc_dependency() -> None:
    source_root = Path(__file__).parents[1] / "src" / "rltracer"
    for path in source_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import jtc_data_commons" not in source
        assert "from jtc_data_commons" not in source



def test_public_api_comes_from_standalone_package() -> None:
    assert rltracer.RLTracer.__module__ == "rltracer.tracer"
