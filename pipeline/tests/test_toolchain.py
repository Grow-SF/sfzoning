import json
from pathlib import Path

import pytest

from sfzmaps import build, config, toolchain


def test_a_tool_counts_only_when_it_resolves_into_the_pixi_environment(tmp_path, monkeypatch):
    env = tmp_path / ".pixi" / "envs" / "default" / "bin"
    env.mkdir(parents=True)
    (env / "gdalwarp").write_text("")
    link = tmp_path / "bin" / "gdalwarp"
    link.parent.mkdir()
    link.symlink_to(env / "gdalwarp")
    monkeypatch.setenv("CONDA_PREFIX", str(env.parent))
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(tmp_path))
    assert toolchain.env_prefix() == env.parent.resolve()
    assert toolchain.Tool("gdalwarp", str(link), None).from_env           # via a link
    assert not toolchain.Tool("gdalwarp", "/opt/homebrew/bin/gdalwarp", None).from_env
    assert not toolchain.Tool("pdftoppm", None, None).from_env           # missing


def test_outside_pixi_the_repositorys_default_environment_is_expected(monkeypatch):
    monkeypatch.delenv("PIXI_PROJECT_ROOT", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", "/somewhere/else")              # another conda env
    assert toolchain.env_prefix() == (config.PIPELINE_DIR.parent / ".pixi/envs/default").resolve()


def test_build_refuses_tools_outside_the_flake_unless_told(monkeypatch, capsys):
    monkeypatch.setattr(toolchain, "not_from_env", lambda: [toolchain.Tool("pdftoppm", "/opt/homebrew/bin/pdftoppm", "26.10.0")])
    with pytest.raises(SystemExit, match="pixi run build"):
        build.require_toolchain(False)
    build.require_toolchain(True)
    assert "WARNING" in capsys.readouterr().out
    monkeypatch.setattr(toolchain, "not_from_env", lambda: [])
    build.require_toolchain(False)


def test_every_stage_depends_on_the_pinned_toolchain():
    built = [st for st in build.stages().values() if not st.frozen]     # refdata is never built
    assert built and all({"../pixi.toml", "../pixi.lock"} <= set(build.stage_deps(st)) for st in built)


def test_render_check_compares_fresh_renders_with_the_recorded_hashes(tmp_path, monkeypatch):
    from sfzmaps import pages
    from sfzmaps.pages import Page
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    ps = []
    for sid, sha in (("p1", toolchain.hashlib.sha256(b"same").hexdigest()), ("p2", "recorded-elsewhere")):
        src = tmp_path / f"{sid}.pdf"
        src.write_bytes(b"%PDF")
        ps.append(Page("1964", sid, src, 0))
        pj = tmp_path / "work" / "1964" / sid / "page.json"
        pj.parent.mkdir(parents=True)
        pj.write_text(json.dumps({"kind": "map", "page_sha256": sha}))
    monkeypatch.setattr(pages, "discover", lambda raw: ps)
    written = []

    def render(page, out):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"same")
        written.append(Path(out))
        return out

    monkeypatch.setattr(pages, "render", render)
    n, differ = toolchain.check_renders(5)
    assert n == 2 and differ == ["1964/p2"]
    assert all(not p.exists() for p in written)                          # a temp folder, gone
    assert not any((tmp_path / "work").rglob("*.png"))                    # nothing under data/
