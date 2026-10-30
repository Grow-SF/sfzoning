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
