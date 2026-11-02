import hashlib
import json

import pytest

from sfzmaps import answers, config, llm

K1 = "llm/" + "1" * 64 + ".json"
K2 = "llm/tiles/" + "2" * 64 + ".json"


def test_sidecar_round_trip_sorted_and_deduplicated(tmp_path):
    art = tmp_path / "labels.json"
    answers.write(art, [K2, K1, K1])
    assert answers.sidecar(art).name == "labels.json.answers"
    assert answers.read(art) == sorted([K1, K2])
    assert answers.read(tmp_path / "none.json") is None


@pytest.mark.parametrize("bad", ["../../.env", "llm/../../pipeline/.env", "llm/abc.json",
                                 "/etc/passwd", "llm/tiles/" + "g" * 64 + ".json"])
def test_only_answer_cache_paths_are_accepted(tmp_path, bad):
    with pytest.raises(ValueError):
        answers.write(tmp_path / "x.json", [K1, bad])
    answers.sidecar(tmp_path / "y.json").write_text(bad + "\n")
    with pytest.raises(ValueError):
        answers.read(tmp_path / "y.json")


def test_baseline_covers_exactly_the_listed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "PIPELINE_DIR", tmp_path)
    art = tmp_path / "data" / "work" / "1948" / "6a" / "labels.json"
    art.parent.mkdir(parents=True)
    art.write_text("old")
    assert not answers.covered(art)
    (tmp_path / "cache-baseline.json").write_text(json.dumps(
        {"work/1948/6a/labels.json": hashlib.md5(b"old").hexdigest()}))
    assert answers.covered(art)
    art.write_text("new")                                 # changed since: not covered any more
    assert not answers.covered(art)
    answers.write(art, [K1])
    assert answers.covered(art)


def test_consume_folds_an_artifacts_answers_into_the_current_recording(tmp_path):
    art = tmp_path / "district_legend.json"
    answers.write(art, [K1])
    with llm.recording() as keys:
        answers.consume(art)
        answers.record([K2])
    assert keys == {K1, K2}


def test_remove_takes_the_sidecar_with_it(tmp_path):
    art = tmp_path / "labels.json"
    art.write_text("[]")
    answers.write(art, [K1])
    answers.remove(art)
    assert not art.exists() and not answers.sidecar(art).exists()


def test_baseline_lists_only_artifacts_without_sidecars_and_forgets_changed_ones(tmp_path, monkeypatch):
    for n in ("DATA_DIR", "WORK_DIR", "OUT_DIR"):
        monkeypatch.setattr(config, n, tmp_path / "data" / n.lower().removesuffix("_dir"))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "data" / "work")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "data" / "out")
    monkeypatch.setattr(config, "PIPELINE_DIR", tmp_path)
    old = tmp_path / "data" / "work" / "1948" / "6a" / "labels.json"
    new = tmp_path / "data" / "work" / "1948" / "7" / "labels.json"
    partial = tmp_path / "data" / "work" / "1948" / "districts_raw.partial.parquet"
    for p in (old, new, partial):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    answers.write(new, [K1])
    assert answers.write_baseline(answers.all_artifacts()) == 1
    assert answers.covered(old) and list(json.loads(answers.baseline_path().read_text())) == ["work/1948/6a/labels.json"]
    old.write_text("changed")
    assert answers.write_baseline([]) == 0 and not answers.covered(old)
