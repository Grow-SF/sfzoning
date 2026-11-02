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
    assert list(table)[-1] == "shared" and set(table["shared"].upstream) == set(table) - {"shared"}


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
    assert build.plan_run({"edition-1948"}, None, t) == ["edition-1948", "shared"]
    assert build.plan_run({"edition-1948", "today"}, ["edition-1948"], t) == ["edition-1948"]
    assert build.plan_run(set(), ["today"], t) == []
    assert build.plan_run(set(), ["today"], t, force=True) == ["today"]
    assert build.plan_run({"today"}, ["shared"], t) == ["today", "shared"]
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


def test_stage_fails_when_a_sheet_newly_errors(tmp_path, monkeypatch):
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
