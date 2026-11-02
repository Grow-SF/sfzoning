import json
from pathlib import Path

import pytest

from sfzmaps import build, config, toolchain


def test_a_tool_counts_as_nix_only_when_it_resolves_into_the_store(tmp_path):
    store = tmp_path / "nix" / "store" / "abc-gdal" / "bin"
    store.mkdir(parents=True)
    (store / "gdalwarp").write_text("")
    link = tmp_path / "bin" / "gdalwarp"
    link.parent.mkdir()
    link.symlink_to(store / "gdalwarp")
    old = toolchain.NIX_STORE
    try:
        toolchain.NIX_STORE = str(tmp_path / "nix" / "store") + "/"
        assert toolchain.Tool("gdalwarp", str(link), None).from_nix          # via a profile link
        assert not toolchain.Tool("gdalwarp", "/opt/homebrew/bin/gdalwarp", None).from_nix
        assert not toolchain.Tool("pmtiles", None, None).from_nix              # missing
    finally:
        toolchain.NIX_STORE = old


def test_build_refuses_tools_outside_the_flake_unless_told(monkeypatch, capsys):
    monkeypatch.setattr(toolchain, "not_from_nix", lambda: [toolchain.Tool("pdftoppm", "/opt/homebrew/bin/pdftoppm", "26.10.0")])
    with pytest.raises(SystemExit, match="nix develop"):
        build.require_toolchain(False)
    build.require_toolchain(True)
    assert "WARNING" in capsys.readouterr().out
    monkeypatch.setattr(toolchain, "not_from_nix", lambda: [])
    build.require_toolchain(False)


def test_every_stage_depends_on_the_flake():
    assert all("../flake.nix" in build.stage_deps(st) for st in build.stages().values())


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
