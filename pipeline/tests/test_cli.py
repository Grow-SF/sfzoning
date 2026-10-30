import types
from pathlib import Path

from sfzmaps import runner
from sfzmaps.pages import Page


def test_cli_run_all_continues_past_a_failing_series(monkeypatch):
    """M3: one series failing to publish does not stop `run --all`; failures are summarized."""
    import subprocess

    from typer.testing import CliRunner

    from sfzmaps import cli, pages, stitch
    from sfzmaps.sources import datasf

    monkeypatch.setattr(datasf, "load_reference", lambda: None)
    monkeypatch.setattr(pages, "discover", lambda: [Page(s, "1", Path("x"), None) for s in ("a", "b", "c")])
    ran, published = [], []
    monkeypatch.setattr(runner, "run_series", lambda s, **k: ran.append(s) or {})

    def publish(s, *a, **k):
        if s == "b":
            raise subprocess.CalledProcessError(1, ["gdalbuildvrt"])
        published.append(s)
    monkeypatch.setattr(cli, "_publish", publish)
    res = CliRunner().invoke(cli.app, ["run", "--all"])
    assert ran == ["a", "b", "c"] and published == ["a", "c"]
    assert res.exit_code == 1 and "1 series failed" in res.output and "b: CalledProcessError" in res.output

    # `tiles` reports any error as one line, not only RuntimeError
    monkeypatch.setattr(stitch, "load_scores", lambda s: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["gdal"])))
    monkeypatch.setattr(cli, "_publish", lambda *a: stitch.load_scores("b"))
    res = CliRunner().invoke(cli.app, ["tiles", "--year", "b"])
    assert res.exit_code == 1 and "error: CalledProcessError" in res.output


def test_cli_districts_series_and_all(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli, reviewpage
    from sfzmaps.districts import assemble
    from sfzmaps.districts.series import SERIES

    ran, pages = [], []
    monkeypatch.setattr(assemble, "run_series",
                        lambda s, **k: ran.append((s, k.get("sheets"))) or tmp_path)
    monkeypatch.setattr(reviewpage, "write", lambda s: pages.append(s) or tmp_path / "review.html")
    res = CliRunner().invoke(cli.app, ["districts", "--series", "1948", "--sheet", "1"])
    assert res.exit_code == 0, res.output
    assert ran == [("1948", ["1"])] and pages == []      # partial: no review page

    ran.clear(), pages.clear()
    res = CliRunner().invoke(cli.app, ["districts", "--series", "today"])
    assert res.exit_code == 0 and ran == [("today", None)] and pages == []

    ran.clear(), pages.clear()
    res = CliRunner().invoke(cli.app, ["districts", "--all"])
    assert res.exit_code == 0, res.output
    names = [s for s, _ in ran]
    assert set(names) == {s for s, i in SERIES.items() if i.role != "skip"} | {"today"}
    assert "2000-digital-zoning" in pages and "2000-digital-height" not in pages

    res = CliRunner().invoke(cli.app, ["districts", "--series", "2000-digital-su"])
    assert res.exit_code != 0
    res = CliRunner().invoke(cli.app, ["districts"])
    assert res.exit_code != 0


def test_cli_districts_quota_aborts_with_resume_hint(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli, llm, reviewpage
    from sfzmaps.districts import assemble

    ran = []

    def run(s, **k):
        ran.append(s)
        if s == "1960":
            raise llm.LLMQuotaError("Organization spend limit reached")
        return tmp_path

    monkeypatch.setattr(assemble, "run_series", run)
    monkeypatch.setattr(reviewpage, "write", lambda s: tmp_path / "review.html")
    res = CliRunner().invoke(cli.app, ["districts", "--all"])
    assert res.exit_code == 2
    assert ran[-1] == "1960" and "1964" not in ran          # the loop stopped
    assert "series failed" not in res.output
    assert "resume" in res.output and "sfzmaps districts --all" in res.output
    assert "1960" in res.output.split("not finished:")[1]


def test_cli_run_auth_aborts_with_resume_hint(monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import cli, llm, pages, runner
    from sfzmaps.sources import datasf

    monkeypatch.setattr(datasf, "load_reference", lambda: None)
    monkeypatch.setattr(pages, "discover", lambda: [types.SimpleNamespace(series=s) for s in ("1921", "1948")])
    ran = []

    def rs(s, **k):
        ran.append(s)
        raise llm.LLMAuthError("OPENAI_API_KEY missing or invalid")

    monkeypatch.setattr(runner, "run_series", rs)
    res = CliRunner().invoke(cli.app, ["run", "--all"])
    assert res.exit_code == 2 and ran == ["1921"]
    assert "series failed" not in res.output and "sfzmaps run --all" in res.output


def test_cli_republish_and_cache_only_never_reach_network(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli, llm, reviewpage
    from sfzmaps.districts import assemble

    seen = []
    monkeypatch.setattr(assemble, "republish", lambda s: seen.append(("re", s, llm.CACHE_ONLY)) or tmp_path)
    monkeypatch.setattr(assemble, "run_series",
                        lambda s, **k: seen.append(("run", s, k.get("cache_only"))) or tmp_path)
    monkeypatch.setattr(reviewpage, "write", lambda s: tmp_path / "review.html")
    res = CliRunner().invoke(cli.app, ["districts", "--republish", "--all"])
    assert res.exit_code == 0, res.output
    assert {s for _, s, _ in seen} == {"1921", "1929", "1935", "1948", "1960", "1964", "1968", "1972",
                                       "1975", "1980", "1980-1990-zoning", "1990-1996-zoning",
                                       "1996-2000-zoning", "2000-digital-zoning", "today"}
    assert all(k == "re" and co for k, _, co in seen)
    seen.clear()
    res = CliRunner().invoke(cli.app, ["districts", "--series", "1960", "--cache-only"])
    assert res.exit_code == 0 and seen == [("run", "1960", True)]


def test_cli_districts_status(monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import cli
    from sfzmaps.districts import assemble

    monkeypatch.setattr(assemble, "status_rows", lambda: [
        {"series": "1960", "use": {"reader": "B", "region_read": True, "git": "abc1234",
                                   "read_at": "2026-10-30T04:00"}, "height": None,
         "has_height": False, "needs_rerun": False, "rerun": []}])
    res = CliRunner().invoke(cli.app, ["districts", "--status"])
    assert res.exit_code == 0, res.output
    assert "1960" in res.output and "region-read abc1234" in res.output and "no" in res.output
