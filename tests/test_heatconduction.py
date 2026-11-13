from pathlib import Path
from typing import Any

import pytest

from engibench.problems.heatconduction2d import shared
from engibench.utils import container


def test_run_container_script_mounts_jit_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(container, "run", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    shared.run_container_script("image", Path(shared.__file__), (), "out.txt")

    jit_cache = tmp_path / ".cache" / "engibench" / "dijitso"
    assert jit_cache.is_dir()
    assert (str(jit_cache), "/home/fenics/.cache/dijitso") in calls[0]["mounts"]
    assert calls[0]["env"] == {"DIJITSO_CACHE_DIR": "/home/fenics/.cache/dijitso"}
