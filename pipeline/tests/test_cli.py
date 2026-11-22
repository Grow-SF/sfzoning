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
    assert res.exit_code == cli.PAID_READ_EXIT
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
    assert res.exit_code == cli.PAID_READ_EXIT and ran == ["1921"]
    assert "series failed" not in res.output and "sfzmaps run --all" in res.output


def test_cli_republish_and_cache_only_never_reach_network(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli, llm, reviewpage
    from sfzmaps.districts import assemble

    seen = []
    monkeypatch.setattr(assemble, "republish", lambda s, **k: seen.append(("re", s, llm.CACHE_ONLY)) or tmp_path)
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


def test_cli_downloads_rejects_unknown_series(monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import build, cli, downloads

    monkeypatch.setattr(build, "require_toolchain", lambda s: None)
    monkeypatch.setattr(downloads, "catalog", lambda: [])
    res = CliRunner().invoke(cli.app, ["downloads", "--series", "nope"])
    assert res.exit_code != 0 and "nope" in res.output


def test_cli_downloads_listed(monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import cli

    res = CliRunner().invoke(cli.app, ["--help"])
    assert "downloads" in res.output
    res = CliRunner().invoke(cli.app, ["downloads", "--help"], env={"NO_COLOR": "1", "TERM": "dumb"})
    import re
    assert res.exit_code == 0 and "--series" in re.sub(r"\x1b\[[0-9;]*m", "", res.output)


def _frame_stubs(monkeypatch, tmp_path, stale=None, test_sheets=(("s", "2"),)):
    """Everything `frame` calls, recording its calls ("turn" when a GPU turn is entered; the run's steps enter the
    turn they are given, as the real ones do around their GPU work); the output folder is `tmp_path`.  `stale` is
    what `model_stale` says (None: the model is current); `test_sheets` the test set (the training set is s/1)."""
    import contextlib

    from sfzmaps import gpu
    from sfzmaps.frame import data as fd
    from sfzmaps.frame import look
    from sfzmaps.frame import run as fr

    calls = []
    monkeypatch.setattr(fd, "OUT", tmp_path)
    monkeypatch.setattr(fd, "labelled_sheets", lambda *a, **k: [("s", "1")])
    monkeypatch.setattr(fd, "test_sheets", lambda *a, **k: list(test_sheets))

    @contextlib.contextmanager
    def turn(*a, **k):
        calls.append("turn")
        yield
    monkeypatch.setattr(gpu, "gpu_turn", turn)

    def step(name):
        def run_step(*a, turn=None, **k):
            if turn is not None:
                with turn():
                    pass
            calls.append(name)
            return {}
        return run_step
    for name in ("fit", "score_dev", "score_test", "predict_all"):
        monkeypatch.setattr(fr, name, step(name))
    monkeypatch.setattr(fr, "model_stale", lambda out: calls.append("model_stale") or stale)
    monkeypatch.setattr(fr, "check_disk", lambda out: calls.append("check_disk"))
    monkeypatch.setattr(fr, "format_split", lambda r: "split table")
    monkeypatch.setattr(look, "series_mosaics", lambda out: calls.append(("mosaics", out)) or [])
    monkeypatch.setattr(look, "closeups", lambda out: calls.append("closeups"))
    monkeypatch.setattr(look, "set_mosaic", lambda out, name: calls.append(("set mosaic", name)) or out / name)
    return calls


def test_cli_frame_test_refuses_an_overlap_before_any_gpu_work(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli

    for flags in (["--test"], ["--dev"], [], ["--predict-only"]):
        calls = _frame_stubs(monkeypatch, tmp_path, stale="model.json is missing", test_sheets=[("s", "1")])
        res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu", *flags])
        assert res.exit_code == 1 and "s/1" in res.output, flags
        assert calls == [], flags


def test_cli_frame_dev_and_test_fit_first_when_the_model_is_stale(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli
    from sfzmaps.frame import run as fr

    for flag, step, where in (("--dev", "score_dev", fr.DEV_DIR), ("--test", "score_test", fr.TEST_DIR)):
        calls = _frame_stubs(monkeypatch, tmp_path, stale="model.json is missing")
        res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu", flag])
        assert res.exit_code == 0, res.output
        assert "split table" in res.output and "model.json is missing" in res.output
        assert calls == ["model_stale", "turn", "fit", step, ("mosaics", tmp_path / where), ("set mosaic", where)]
        calls = _frame_stubs(monkeypatch, tmp_path, stale=None)
        res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu", flag])
        assert res.exit_code == 0, res.output
        assert calls == ["model_stale", step, ("mosaics", tmp_path / where),
                         ("set mosaic", where)]                                        # current: no fit, no GPU


def test_cli_frame_plain_fits_when_stale_then_predicts(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli

    calls = _frame_stubs(monkeypatch, tmp_path, stale="the training sheets changed")
    res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu"])
    assert res.exit_code == 0, res.output
    assert calls == ["check_disk", "model_stale", "turn", "fit", "turn", "predict_all", ("mosaics", tmp_path),
                     "closeups"]
    calls = _frame_stubs(monkeypatch, tmp_path, stale=None)
    res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu"])
    assert res.exit_code == 0, res.output
    assert calls == ["check_disk", "model_stale", "turn", "predict_all", ("mosaics", tmp_path), "closeups"]


def test_cli_frame_predict_only_refuses_a_stale_model_before_any_gpu_work(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli

    calls = _frame_stubs(monkeypatch, tmp_path, stale="the reducer changed")
    res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu", "--predict-only"])
    assert res.exit_code == 1 and "the reducer changed" in res.output and "sfzmaps frame" in res.output
    assert calls == ["check_disk", "model_stale"]


def test_cli_frame_full_disk_is_an_error(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli
    from sfzmaps.frame import run as fr

    calls = _frame_stubs(monkeypatch, tmp_path)

    def full(out):
        raise fr.DiskFull("only 3.0 GB free under x; predicting every page needs 20 GB")
    monkeypatch.setattr(fr, "check_disk", full)
    for flags in ([], ["--predict-only"]):
        res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu", *flags])
        assert res.exit_code == 1 and "20 GB" in res.output and calls == []
    res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu", "--test"])   # no prediction: no check
    assert res.exit_code == 0 and calls == ["model_stale", "score_test", ("mosaics", tmp_path / fr.TEST_DIR),
                                            ("set mosaic", fr.TEST_DIR)]


def test_cli_frame_flags_exclude_each_other(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from sfzmaps import cli

    calls = _frame_stubs(monkeypatch, tmp_path)
    for flags in (["--dev", "--test"], ["--dev", "--predict-only"], ["--test", "--predict-only"]):
        res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu", *flags])
        assert res.exit_code != 0, flags
    assert calls == []
    res = CliRunner().invoke(cli.app, ["frame", "--device", "cpu", "--held-out"])
    assert res.exit_code != 0 and calls == []                                  # retired
