import json

from sfzmaps import fresh


def test_versions_are_part_of_the_fingerprint_only_when_given(tmp_path):
    src, out = tmp_path / "in.txt", tmp_path / "out.txt"
    src.write_text("a")
    out.write_text("made")
    fresh.record(out, [src])
    assert set(json.loads(fresh.sidecar(out).read_text())) == {"version", "inputs"}    # sidecars made before
    assert fresh.is_fresh(out, [src])
    fresh.record(out, [src], versions={"torch": "2.15.0"})
    assert fresh.is_fresh(out, [src], versions={"torch": "2.15.0"})
    assert not fresh.is_fresh(out, [src], versions={"torch": "2.16.0"})
    assert not fresh.is_fresh(out, [src])
