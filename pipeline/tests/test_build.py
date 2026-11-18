import json
import threading
import time

import pytest

from sfzmaps import build, config
from sfzmaps.districts.series import SERIES


def test_every_series_is_in_exactly_one_stage():
    table = build.stages()
    owned = [s for st in table.values() for s in st.series]
    assert sorted(owned) == sorted(SERIES) and len(owned) == len(set(owned))
    # shared comes last and needs the old stages only: not the saved reads' images (no output is made
    # from them), not the warp stages (the old pipeline's merge doesn't read them) and not the solver's
    assert list(table)[-1] == "shared" and set(table["shared"].upstream) == {
        n for n in table if n not in ("shared", "saved-reads") and not n.startswith("warp-")
        and not table[n].command}


def test_a_period_reads_heights_before_zoning_and_never_special_use():
    st = build.stages()["period-1980-1990"]
    assert st.districts == ("1980-1990-height", "1980-1990-zoning")
    assert build.stage_commands("period-1980-1990")[-2:] == [
        ["districts", "--series", "1980-1990-height"], ["districts", "--series", "1980-1990-zoning"]]
    assert build.stage_commands("edition-1948") == [
        ["run", "--year", "1948"], ["paper", "--year", "1948"], ["districts", "--series", "1948"]]
    assert build.stage_commands("shared") == [["districts", "--shared"]]


def test_committed_dvc_yaml_is_the_generated_one():
    assert build.dvc_yaml_path().read_text() == build.dvc_yaml(), \
        "run `uv run sfzmaps build --write-dvc-yaml`"


def test_dvc_yaml_is_the_same_with_no_data(tmp_path, monkeypatch):
    # a fresh clone writes (and checks) dvc.yaml before anything is pulled
    from sfzmaps import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "data" / "work")
    assert build.dvc_yaml() == build.dvc_yaml_path().read_text()


def test_stage_deps_are_per_series_and_skip_the_editor():
    st = build.stages()["edition-1948"]
    deps = build.stage_deps(st)
    assert "inputs/1948" in deps and "inputs/1921" not in deps and "data/raw" in deps
    assert not any(d.startswith("src/sfzmaps/editor") or d.endswith("build.py") for d in deps)
    assert build.stage_outs(st) == ["data/work/1948", "data/out/1948"]
    assert "data/out/1975/districts" in build.stage_deps(build.stages()["shared"])
    assert "data/out/1980-1990-height/districts" not in build.stage_deps(build.stages()["shared"])


def test_status_keys_with_or_without_a_path_prefix():
    j = json.dumps({"pipeline/dvc.yaml:edition-1921": [], "shared": [], "raw.dvc": []})
    assert build.stale_stages(j) == {"edition-1921", "shared"}
    assert build.stale_stages("{}") == set() == build.stale_stages("")


def test_plan_adds_downstream_and_limits_to_targets():
    t = build.stages()
    assert build.plan_run({"warp-1948"}, None, t) == ["warp-1948"]
    assert build.plan_run({"edition-1948"}, None, t) == []        # the old stages are frozen
    assert build.plan_run({"warp-1948", "warp-1975"}, ["warp-1948"], t) == ["warp-1948"]
    assert build.plan_run(set(), ["warp-1948"], t) == []
    assert build.plan_run(set(), ["warp-1948"], t, force=True) == ["warp-1948"]
    with pytest.raises(ValueError):
        build.plan_run(set(), ["edition-1999"], t)


def _table():
    S = build.Stage
    return {"a": S("a", (), ()), "b": S("b", (), ()), "c": S("c", (), ()),
            "z": S("z", (), (), ("a", "b", "c"))}


def test_build_runs_in_parallel_up_to_jobs_and_downstream_after_upstream():
    live, peak, seen = 0, 0, []
    lock = threading.Lock()

    def run(n):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
            seen.append(("start", n))
        time.sleep(0.05)
        with lock:
            live -= 1
            seen.append(("end", n))

    commits = []
    res = build.run_build(["a", "b", "c", "z"], _table(), 2, run_stage=run, commit=commits.append)
    assert res.ok and sorted(res.ran) == ["a", "b", "c", "z"] and peak == 2
    assert seen.index(("start", "z")) > max(seen.index(("end", n)) for n in "abc")
    assert sorted(commits) == ["a", "b", "c", "z"]


def test_a_failed_stage_is_not_recorded_and_its_downstream_is_skipped():
    def run(n):
        if n == "b":
            raise build.StageFailed("boom")

    commits = []
    res = build.run_build(["a", "b", "c", "z"], _table(), 3, run_stage=run, commit=commits.append)
    assert not res.ok and set(res.failed) == {"b"} and res.skipped == ["z"]
    assert sorted(commits) == ["a", "c"]


def test_inputs_changing_mid_build_stop_recording():
    calls = iter([None, "code or inputs changed during the build"])
    commits = []
    res = build.run_build(["a", "b", "z"], _table(), 1, run_stage=lambda n: None,
                          commit=commits.append, guard=lambda: next(calls, "changed"))
    assert commits == ["a"] and "b" in res.failed and res.skipped == ["z"]


def _chain():
    """a -> b -> c, and d beside them"""
    S = build.Stage
    return {"a": S("a", (), ()), "b": S("b", (), (), ("a",)), "c": S("c", (), (), ("b",)), "d": S("d", (), ())}


def test_a_stage_planned_only_for_its_upstream_is_checked_again_once_that_is_recorded():
    events = []
    still_stale = {"b": False, "c": True}         # a ran and changed nothing b reads; c changed on its own

    def recheck(n):
        events.append(("recheck", n))
        return still_stale[n]

    res = build.run_build(["a", "b", "c"], _chain(), 1, run_stage=lambda n: events.append(("run", n)),
                          commit=lambda n: events.append(("commit", n)), recheck=recheck, must_run={"a"})
    # b is asked again only after a is recorded; up to date, it doesn't run, and c (its downstream) is asked in turn
    assert events == [("run", "a"), ("commit", "a"), ("recheck", "b"), ("recheck", "c"), ("run", "c"),
                      ("commit", "c")]
    assert res.ok and res.ran == ["a", "c"] and res.fresh == ["b"] and not res.skipped


def test_stale_stages_always_run_and_a_clean_upstream_frees_its_downstream():
    asked = []
    res = build.run_build(["a", "b", "c", "d"], _chain(), 2, run_stage=lambda n: None, commit=lambda n: None,
                          recheck=lambda n: asked.append(n) or False, must_run={"a", "c", "d"})
    assert asked == ["b"] and sorted(res.ran) == ["a", "c", "d"] and res.fresh == ["b"] and res.ok
    # without must_run (and without recheck) every planned stage runs, as before
    res = build.run_build(["a", "b", "c"], _chain(), 1, run_stage=lambda n: None, commit=lambda n: None)
    assert res.ran == ["a", "b", "c"] and res.fresh == []


def test_a_stage_whose_upstream_failed_is_skipped_not_checked_again():
    def run(n):
        if n == "a":
            raise build.StageFailed("boom")

    res = build.run_build(["a", "b", "c"], _chain(), 1, run_stage=run, commit=lambda n: None,
                          recheck=lambda n: pytest.fail("not asked"), must_run={"a"})
    assert set(res.failed) == {"a"} and res.skipped == ["b", "c"] and res.fresh == []


class _Done:
    def __init__(self, returncode, stdout):
        self.returncode, self.stdout, self.stderr = returncode, stdout, ""


def test_the_second_status_asks_dvc_about_that_stage_alone():
    calls = []

    def dvc(*args, check=True):
        calls.append(args)
        return {"solver-lots": _Done(0, "{}"),
                "solver-prints": _Done(0, json.dumps({"dvc.yaml:solver-prints": [
                    {"changed deps": {"data/solver/lots": "modified"}}]})),
                "solver-labels": _Done(1, "")}[args[-1]]

    assert build.still_stale("solver-lots", dvc) == (False, [])
    assert build.still_stale("solver-prints", dvc) == (True, ["data/solver/lots"])
    assert build.still_stale("solver-labels", dvc) == (True, [])       # a status that fails: run it
    assert calls == [("status", "--json", n) for n in ("solver-lots", "solver-prints", "solver-labels")]


def test_a_build_asks_dvc_again_about_the_stages_after_a_stale_one(monkeypatch, tmp_path, capsys):
    import types
    table = _chain()
    yaml = tmp_path / "dvc.yaml"
    yaml.write_text("stages\n")
    for name, value in (("stages", lambda: table), ("dvc_yaml", lambda: "stages\n"), ("dvc_yaml_path", lambda: yaml),
                        ("require_toolchain", lambda system_tools: None), ("git_dirty", lambda *a: []),
                        ("git_head", lambda: "head"), ("prepare", lambda: None)):
        monkeypatch.setattr(build, name, value)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    asked, ran = [], []

    def dvc(*args, check=True):
        asked.append(args)
        if args == ("status", "--json"):                       # at the start: only a is stale
            return _Done(0, json.dumps({"dvc.yaml:a": [{"changed deps": {"inputs/a": "modified"}}]}))
        if args[:2] == ("status", "--json"):                   # asked again: b unchanged, c changed on its own
            return _Done(0, json.dumps({"c": [{"changed deps": {"inputs/c": "modified"}}]} if args[2] == "c" else {}))
        return _Done(0, "")                                    # dvc commit

    monkeypatch.setattr(build, "dvc", dvc)
    monkeypatch.setattr(build, "subprocess", types.SimpleNamespace(
        STDOUT=-2, run=lambda argv, **kw: ran.append((argv[-1], kw["env"]["SFZMAPS_DVC_CHANGED_DEPS"])) or _Done(0, "")))
    assert build.build(None, 1, dry_run=False, force=False) == 0
    # b is asked again once a is recorded, and not run; c is asked once b counts as done, and runs with its own
    # changed deps from the second status
    assert asked == [("status", "--json"), ("commit", "--force", "a"), ("status", "--json", "b"),
                     ("status", "--json", "c"), ("commit", "--force", "c")]
    assert ran == [("a", '["inputs/a"]'), ("c", '["inputs/c"]')]
    out = capsys.readouterr().out
    assert "b, c only if still stale" in out and "up to date after its upstreams: b" in out


def _stage_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    sj = tmp_path / "work" / "1948" / "s1" / "score.json"
    sj.parent.mkdir(parents=True)
    sj.write_text(json.dumps({"reason": "holdout_median"}))
    return sj


def test_stage_runs_its_commands_at_zero_dollars(tmp_path, monkeypatch):
    _stage_env(tmp_path, monkeypatch)
    got = []
    build.run_stage("edition-1948", run=lambda argv, env: (got.append((argv, env["SFZMAPS_MAX_USD"])) or (0, "")))
    assert [a for a, _ in got] == build.stage_commands("edition-1948") and {u for _, u in got} == {"0"}


def test_stage_fails_while_any_sheet_is_in_error_even_on_a_retry(tmp_path, monkeypatch):
    sj = _stage_env(tmp_path, monkeypatch)

    def run(argv, env):
        if argv[0] == "run":
            sj.write_text(json.dumps({"reason": "error: MemoryError"}))
        return 0, ""

    with pytest.raises(build.StageFailed, match="1948/s1"):
        build.run_stage("edition-1948", run=run)


def test_stage_fails_when_paper_skips_or_a_paid_read_is_needed(tmp_path, monkeypatch):
    _stage_env(tmp_path, monkeypatch)
    with pytest.raises(build.StageFailed, match="skipped"):
        build.run_stage("edition-1948", run=lambda a, e: (0, "1948: skipped (no warped sheets)\n")
                        if a[0] == "paper" else (0, ""))
    with pytest.raises(build.StageFailed) as e:
        build.run_stage("edition-1948", run=lambda a, e: (build.PAID_READ_EXIT, "") if a[0] == "districts" else (0, ""))
    assert e.value.paid_read and "SFZMAPS_MAX_USD" in str(e.value)


def test_zero_dollar_cap_refuses_a_live_call_after_the_cache(monkeypatch):
    import numpy as np
    from pydantic import BaseModel

    from sfzmaps import llm

    class A(BaseModel):
        x: int

    class Client:
        def call(self, *a):
            raise AssertionError("no live call at $0")

    monkeypatch.setenv("SFZMAPS_MAX_USD", "0")
    with pytest.raises(llm.LLMQuotaError):
        llm.ask_json(np.zeros((4, 4, 3), np.uint8), "p", A, client=Client())


def test_one_build_at_a_time(tmp_path):
    with build.build_lock(tmp_path / "l"):
        with pytest.raises(SystemExit, match="another build"):
            with build.build_lock(tmp_path / "l"):
                pass


def test_build_cli_lists_stages_and_checks_dvc_yaml(monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps.cli import app
    r = CliRunner().invoke(app, ["build", "--list"])
    assert r.exit_code == 0 and "period-1980-1990" in r.output and "shared" in r.output
    r = CliRunner().invoke(app, ["build", "--force"])
    assert r.exit_code != 0
    monkeypatch.setattr(build, "dvc_yaml", lambda: "different\n")
    r = CliRunner().invoke(app, ["build", "--dry-run"])
    assert r.exit_code != 0 and "dvc.yaml is out of date" in str(r.exception or r.output)


def test_paid_read_exit_codes_agree():
    from sfzmaps import cli
    assert build.PAID_READ_EXIT == cli.PAID_READ_EXIT != 2      # 2 is click's usage error


def test_a_failed_record_is_reported_not_raised():
    def commit(n):
        if n == "a":
            raise RuntimeError("dvc commit: lock")
    res = build.run_build(["a", "b"], _table(), 2, run_stage=lambda n: None, commit=commit)
    assert "recording it failed" in res.failed["a"] and set(res.ran) == {"a", "b"} and not res.ok


def _saved_reads(tmp_path, images: dict[str, bytes], listed: dict[str, bytes]):
    """an images folder holding `images`, and a manifest naming `listed` (name -> the bytes it expects)"""
    import hashlib
    d = tmp_path / "images"
    for name, b in images.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b)
    m = tmp_path / "images.manifest.tsv"
    m.write_text("image\tsha256\tbytes\n" + "".join(
        f"{n}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\n" for n, b in listed.items()))
    return m, d


def test_saved_reads_stage_is_frozen_owns_the_images_and_nothing_depends_on_it():
    table = build.stages()
    st = table["saved-reads"]
    assert st.frozen and build.stage_outs(st) == ["data/saved-reads/images"]
    assert build.stage_deps(st) == ["saved-reads/images.manifest.tsv"]
    assert not any("saved-reads" in other.upstream for other in table.values())
    assert "src/sfzmaps/savedreads.py" not in build._code_deps()     # changing it restales no other stage


def test_saved_reads_check_names_missing_changed_and_unlisted_images(tmp_path):
    from sfzmaps import savedreads
    m, d = _saved_reads(tmp_path, {"a/1.png": b"one", "a/2.png": b"two!", "b/x.png": b"extra"},
                        {"a/1.png": b"one", "a/2.png": b"two", "a/3.png": b"three"})
    assert sorted(savedreads.check(m, d)) == ["differs from the manifest: a/2.png", "missing: a/3.png",
                                               "not in the manifest: b/x.png"]
    m, d = _saved_reads(tmp_path / "ok", {"a/1.png": b"one"}, {"a/1.png": b"one"})
    assert savedreads.check(m, d) == []


def test_saved_reads_stage_fails_unless_the_images_match(tmp_path, monkeypatch):
    from sfzmaps import savedreads
    m, d = _saved_reads(tmp_path, {"a/1.png": b"one"}, {"a/1.png": b"uno"})
    monkeypatch.setattr(savedreads, "manifest_path", lambda: m)
    monkeypatch.setattr(savedreads, "images_dir", lambda: d)
    with pytest.raises(build.StageFailed, match="differs from the manifest: a/1.png"):
        build.run_stage("saved-reads", run=lambda argv, env: pytest.fail("runs no sfzmaps command"))
    (d / "a/1.png").write_bytes(b"uno")
    build.run_stage("saved-reads", run=lambda argv, env: pytest.fail("runs no sfzmaps command"))


def test_every_series_has_one_warp_stage_after_the_stage_that_places_it():
    table = build.stages()
    names = list(table)
    warps = {n: st for n, st in table.items() if st.warp}
    assert sorted(st.warp for st in warps.values()) == sorted(SERIES)
    for n, st in warps.items():
        assert n == f"warp-{st.warp}" and not st.series and not st.districts and not st.frozen
        owner = [o for o, ost in table.items() if st.warp in ost.series]
        assert len(owner) == 1 and st.upstream == (owner[0], "refdata")
        assert names.index(n) > names.index(owner[0]) and names.index(n) < names.index("shared")


def test_a_warp_stage_reads_its_series_work_and_writes_data_warp():
    st = build.stages()["warp-1948"]
    assert build.stage_commands("warp-1948") == [["warp", "1948"]]
    assert build.stage_deps(st) == build._code_deps() + [
        "pyproject.toml", "uv.lock", "../pixi.toml", "../pixi.lock", "data/ref", "data/work/1948"]
    assert build.stage_outs(st) == ["data/warp/1948"]


def test_the_old_pipeline_stages_are_frozen():
    table = build.stages()
    old = [n for n in table if n.startswith(("edition-", "period-")) or n in ("today", "shared")]
    assert old and all(table[n].frozen for n in old)
    assert not any(st.frozen for n, st in table.items() if n.startswith("warp-"))


def test_a_build_prepares_only_for_the_old_stages():
    t = build.stages()
    assert build.needs_prepare(["warp-1948"], t) is False
    assert build.needs_prepare(["warp-1948", "today"], t) is True


def _command_stage():
    return build.Stage("x", (), (), command=("solver", "page"), reads=("data/warp/2000-digital-zoning",),
                       writes=("data/solver/page",), code=("sfzmaps.solver.paths",))


def test_a_command_stage_depends_on_its_import_closure_and_reads(monkeypatch):
    st = _command_stage()
    deps = build.stage_deps(st)
    for d in ("src/sfzmaps/solver/paths.py", "src/sfzmaps/config.py", "pyproject.toml", "uv.lock",
              "data/warp/2000-digital-zoning"):
        assert d in deps
    assert not any(d.startswith("src/sfzmaps/districts") or d.endswith("sheetwarp/probabilistic.py")
                   for d in deps)
    assert not any(d.endswith("build.py") for d in deps)
    assert build.stage_outs(st) == ["data/solver/page"]
    monkeypatch.setattr(build, "stages", lambda: {"x": st})
    assert build.stage_commands("x") == [["solver", "page"]]


def test_the_gpus_turns_are_in_no_command_stages_code_deps():
    from sfzmaps import fresh
    # the detector's stage imports sfzmaps.gpu (its training takes a turn on the GPU) ...
    assert any(p.name == "gpu.py" for p in fresh.code_files(["sfzmaps.solver.textnet.stage"]))
    # ... but the turns decide when it computes, never what: an edit to gpu.py reruns no stage
    for name, st in build.stages().items():
        if st.command:
            assert "sfzmaps.gpu" in st.cut and "src/sfzmaps/gpu.py" not in build.stage_deps(st), name
    # a stage that cuts nothing has it
    st = build.Stage("x", (), (), command=("solver", "text"), code=("sfzmaps.solver.textnet.stage",), cut=())
    assert "src/sfzmaps/gpu.py" in build.stage_deps(st)


def test_the_solver_packages_hold_no_module_list():
    # every solver stage imports both packages: an edit to their docstrings would rerun all of them (the detector's
    # training too), so they are one line each and the module list is in the README
    from sfzmaps import config
    for init in ("solver/__init__.py", "solver/textnet/__init__.py"):
        text = (config.PIPELINE_DIR / "src" / "sfzmaps" / init).read_text()
        assert len(text.splitlines()) == 1 and text.startswith('"""'), init


def test_the_page_and_the_lots_read_only_the_inputs_file_they_use():
    from sfzmaps.solver import paths, table
    t = build.stages()
    for name, used in (("solver-page", f"{paths.CUTS_FILE.parent.name}/{paths.CUTS_FILE.name}"),
                       ("solver-lots", f"{paths.TABLE_DIR.name}/{table.TABLE_FILE}")):
        inputs = [d for d in build.stage_deps(t[name]) if d.startswith("inputs/")]
        # a README, PROVENANCE.md or the table's by-eye corrections (read only in scoring) rerun nothing
        assert inputs == [f"inputs/{used}"], name


def test_the_walls_read_what_build_walls_reads_and_follow_the_one_detector():
    t = build.stages()
    st = t["solver-walls"]
    assert build.stage_commands("solver-walls") == [["solver", "walls"]]
    assert set(st.upstream) == {"solver-labels", "solver-text"}
    deps = build.stage_deps(st)
    for d in ("data/solver/page", "data/solver/lots", "data/solver/labels/strokes.npz",
              "data/solver/prints/terra_boxes.json", "data/solver/text/chance",
              "data/work/2000-digital-zoning",                  # the sheets' pages, where the text chance is read
              "data/warp/2000-digital-zoning", "src/sfzmaps/solver/walls.py", "src/sfzmaps/solver/wallevidence.py",
              "src/sfzmaps/solver/textgrid.py"):
        assert d in deps, d
    assert [d for d in deps if d.startswith("inputs/")] == ["inputs/merged-page/2000-digital-zoning.json"]
    # the held-out detectors' chance is an experiment's (`walls --chance held-out`), never the stage's
    assert "solver-text-reader" not in st.upstream and not any("text-reader" in d for d in deps)
    # not the prints' table, the detector's weights or scores, the rest of the labels, or the solve's code; of the
    # detector's code only what reads its chance (no training, no scoring)
    assert [d for d in deps if d.startswith("data/solver/text")] == ["data/solver/text/chance"]
    assert not any(d.endswith(("textnet/stage.py", "textnet/train.py", "textnet/predict.py", "textnet/model.py",
                               "textnet/scores.py", "textnet/score_stage.py", "textnet/trial.py",
                               "textnet/data.py", "textnet/model.py")) for d in deps)
    assert [d for d in deps if "/textnet/" in d] == ["src/sfzmaps/solver/textnet/boxes.py"]
    assert not any(d in ("data/solver/prints", "data/solver/prints/prints.parquet", "data/solver/labels")
                   or d.endswith(("solver/solve.py", "solver/placement.py", "solver/search.py")) for d in deps)
    assert build.stage_outs(st) == ["data/solver/walls"]


def test_reader_stage_deps():
    from sfzmaps import fresh
    from sfzmaps.solver import paths
    from sfzmaps.solver.codereader import labels
    t = build.stages()
    st = t["solver-text-reader"]
    assert build.stage_commands("solver-text-reader") == [["solver", "text-reader"]]
    assert build.stage_outs(st) == ["data/solver/text-reader"]
    # after the lots, the prints and the training maps' warps
    assert set(st.upstream) == {"solver-lots", "solver-prints", *(f"warp-{s}" for s in paths.TEXT_TRAIN_SERIES)}
    names = list(t)
    assert all(names.index(u) < names.index("solver-text-reader") for u in st.upstream)
    deps = build.stage_deps(st)
    # its code deps are its import closure, without the GPU's turns and the folder swap
    code = [d for d in deps if d.startswith("src/")]
    closure = fresh.code_files(["sfzmaps.solver.textnet.reader_stage"], cut=frozenset(st.cut))
    assert code == sorted(p.relative_to(config.PIPELINE_DIR).as_posix() for p in closure)
    for f in ("textnet/reader_stage.py", "textnet/data.py", "textnet/train.py", "textnet/predict.py",
              "textnet/model.py", "textnet/stage.py", "codereader/labels.py", "grid.py", "paths.py"):
        assert f"src/sfzmaps/solver/{f}" in code, f
    assert "src/sfzmaps/transform.py" in code                   # it reads the placements
    assert not any(d.endswith(("/gpu.py", "solver/runs.py", "textnet/trial.py", "textnet/score_stage.py"))
                   for d in code)
    # of each training sheet its page, Sol's points and its placement (the warp); Terra's boxes and the other passes;
    # the prints, the lots and the owner's answers, for the answered blocks
    rest = [d for d in deps if not d.startswith("src/")]
    for s in paths.TEXT_TRAIN_SERIES:
        for sheet in paths.TEXT_SHEETS[s]:
            for f in (f"data/work/{s}/{sheet}/page.png", f"data/work/{s}/{sheet}/district_labels.json",
                      f"data/warp/{s}/{sheet}/transform.json", f"data/warp/{s}/{sheet}/correction.npz"):
                assert f in rest, f
    for f in ("saved-reads/answers/text-on-every-sheet.jsonl.gz", "saved-reads/answers/experiments-2000/text-positions",
              "data/solver/prints/prints.parquet", "data/solver/prints/terra_boxes.json", "data/solver/lots/lots.npz",
              "data/solver/lots/table.parquet", f"inputs/{labels.CODE_CHECK}/{labels.CASES_FILE}",
              f"inputs/{labels.CODE_CHECK}/{labels.ANSWERS_FILE}", "pyproject.toml", "uv.lock"):
        assert f in rest, f
    assert len(rest) == 2 + 4 * sum(len(paths.TEXT_SHEETS[s]) for s in paths.TEXT_TRAIN_SERIES) + 8
    # nothing of the walls' detector: the reader's detectors are their own
    assert not any(d.startswith("data/solver/text/") or d == "data/solver/text" for d in rest)


def test_text_reader_stages_deps():
    from sfzmaps import fresh
    from sfzmaps.solver import paths
    t = build.stages()
    names = list(t)
    maps = ("2000-digital-zoning", "1980-1990-zoning", "1990-1996-zoning")
    crops, train, read = t["solver-text-crops"], t["solver-text-train"], t["solver-text-read"]
    assert build.stage_commands("solver-text-crops") == [["solver", "text-crops"]]
    assert build.stage_commands("solver-text-train") == [["solver", "text-train"]]
    assert build.stage_commands("solver-text-read") == [["solver", "text-read"]]
    assert set(crops.upstream) == {"solver-lots", "solver-prints", "solver-text-reader", *(f"warp-{s}" for s in maps)}
    assert train.upstream == ("solver-text-crops",)
    assert set(read.upstream) == {"solver-text-crops", "solver-text-train", "solver-boxes"}
    for st in (crops, train, read):
        assert all(names.index(u) < names.index(st.name) for u in st.upstream)
        code = [d for d in build.stage_deps(st) if d.startswith("src/")]
        closure = fresh.code_files(list(st.code), cut=frozenset(st.cut))
        assert code == sorted(p.relative_to(config.PIPELINE_DIR).as_posix() for p in closure)
        assert not any(d.endswith(("/gpu.py", "solver/runs.py")) for d in code)
    # the training reads its crops alone, and only the text reader's training code can retrain it (GPU hours): not
    # the reading maths (ctc.py), which the reading stage alone imports
    assert [d for d in build.stage_deps(train) if not d.startswith("src/")] == ["pyproject.toml", "uv.lock",
                                                                                "data/solver/text-crops"]
    assert [d for d in build.stage_deps(train) if d.startswith("src/sfzmaps/solver/")] == [
        "src/sfzmaps/solver/__init__.py", "src/sfzmaps/solver/paths.py", "src/sfzmaps/solver/textread/__init__.py",
        "src/sfzmaps/solver/textread/model.py", "src/sfzmaps/solver/textread/store.py",
        "src/sfzmaps/solver/textread/train.py"]
    assert "src/sfzmaps/solver/textread/ctc.py" in build.stage_deps(read)
    # the inputs: each scan's pages, Sol's points and placements, Terra's boxes and the other passes, the prints, the
    # lots, the owner's answers and the code reader's detectors' held-out blocks and radius
    rest = [d for d in build.stage_deps(crops) if not d.startswith("src/")]
    for s in maps:
        for sheet in paths.TEXT_SHEETS[s]:
            for f in (f"data/work/{s}/{sheet}/page.png", f"data/work/{s}/{sheet}/district_labels.json",
                      f"data/warp/{s}/{sheet}/transform.json", f"data/warp/{s}/{sheet}/correction.npz"):
                assert f in rest, f
    for f in ("data/solver/text-reader/held.json", "data/solver/prints/prints.parquet", "data/solver/lots/lots.npz"):
        assert f in rest, f
    assert not any(d.startswith("data/solver/text/") for d in rest)
    # the reading: the crops' scoring files, the networks, the boxes and the 2000 pages
    rest = [d for d in build.stage_deps(read) if not d.startswith("src/")]
    assert "data/solver/text-train" in rest and "data/solver/boxes/boxes.parquet" in rest
    assert all(f"data/work/{paths.SERIES}/{sheet}/page.png" in rest for sheet in paths.SHEETS)
    assert not any(d.startswith("data/solver/text-crops/synth") for d in rest)
    assert [build.stage_outs(st) for st in (crops, train, read)] == [["data/solver/text-crops"],
                                                                     ["data/solver/text-train"],
                                                                     ["data/solver/text-read"]]


def test_the_solve_reads_what_the_solve_and_its_report_read():
    t = build.stages()
    st = t["solver-solve"]
    assert build.stage_commands("solver-solve") == [["solver", "solve", "final"]]
    assert "solver-walls" in st.upstream
    deps = build.stage_deps(st)
    for d in ("data/solver/lots", "data/solver/page/ink.png", "data/solver/prints/prints.parquet",
              "data/solver/labels", "data/solver/walls", "data/warp/2000-digital-zoning",
              "src/sfzmaps/solver/solve.py", "src/sfzmaps/solver/placement.py", "src/sfzmaps/solver/holes.py",
              "src/sfzmaps/solver/measures.py", "src/sfzmaps/solver/report.py"):
        assert d in deps, d
    # the hand leader moves and the by-eye corrections the score counts against; not the table file (the lots'
    # table.parquet carries it) or a README
    assert [d for d in deps if d.startswith("inputs/")] == ["inputs/2000-digital-zoning/label_fixes.csv",
                                                            "inputs/zoning-2000/table-corrections.csv"]
    # the walls' and the detector's code reach the solve only through the walls they wrote
    assert not any(d.endswith(("solver/walls.py", "solver/wallevidence.py", "solver/textgrid.py"))
                   or "/textnet/" in d or d.startswith(("data/solver/text-", "data/work/")) for d in deps)
    assert build.stage_outs(st) == ["data/solver/solve"]


def test_needs_prepare_is_false_for_command_stages():
    st = _command_stage()
    assert build.needs_prepare(["x"], {"x": st}) is False


@pytest.mark.parametrize("build_jobs, workers", [(1, 6), (2, 3), (4, 1), (8, 1)])
def test_the_warp_stages_a_build_runs_at_once_share_one_budget_of_workers(monkeypatch, build_jobs, workers):
    monkeypatch.setenv(build.BUILD_JOBS_ENV, str(build_jobs))
    monkeypatch.setattr(build, "WARP_WORKERS", 6)
    got = []
    build.run_stage("warp-1948", run=lambda argv, env: got.append(argv) or (0, ""))
    assert got == [["warp", "1948", "--jobs", str(workers)]]
