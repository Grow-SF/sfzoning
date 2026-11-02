import io
import zipfile

import pytest

from sfzmaps import cachepack, config

K1 = "llm/" + "1" * 64 + ".json"
K2 = "llm/tiles/" + "2" * 64 + ".json"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PIPELINE_DIR", tmp_path / "pipeline")
    for k, body in ((K1, b'{"a":1}'), (K2, b'{"code":"R-1"}')):
        p = tmp_path / "cache" / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    return tmp_path / "cache"


def test_packs_are_deterministic_and_indexed(cache):
    a, b = cachepack.make_pack([K2, K1]), cachepack.make_pack([K1, K2, K1])
    assert a.sha == b.sha and a.data == b.data and a.index_bytes() == b.index_bytes()
    assert set(a.index) == {K1, K2} and cachepack.parse_index(a.index_bytes()) == a.index


def test_apply_is_put_if_absent_and_idempotent(cache, tmp_path, monkeypatch):
    pack = cachepack.make_pack([K1, K2])
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "other")
    assert cachepack.apply_pack(pack.data, pack.sha) == (2, 0)
    assert (tmp_path / "other" / K2).read_bytes() == b'{"code":"R-1"}'
    assert cachepack.apply_pack(pack.data, pack.sha) == (0, 2)


def test_a_corrupt_or_renamed_pack_is_refused(cache):
    pack = cachepack.make_pack([K1])
    with pytest.raises(cachepack.PackError, match="don't match"):
        cachepack.apply_pack(pack.data[:-1] + b"x", pack.sha)


def test_an_answer_that_differs_from_the_stored_one_is_a_conflict(cache, tmp_path, monkeypatch):
    pack = cachepack.make_pack([K1])
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "other")
    (tmp_path / "other" / "llm").mkdir(parents=True)
    (tmp_path / "other" / K1).write_bytes(b'{"a":2}')
    with pytest.raises(cachepack.PackError, match="differ"):
        cachepack.apply_pack(pack.data, pack.sha)
    assert (tmp_path / "other" / K1).read_bytes() == b'{"a":2}'      # never replaced


@pytest.mark.parametrize("name", ["../../evil.json", "llm/../../.env", "llm/x.json"])
def test_a_member_outside_the_answer_cache_is_refused_before_writing(cache, tmp_path, name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(K1, b"{}")
        z.writestr(name, b"x")
    data = buf.getvalue()
    with pytest.raises((cachepack.PackError, ValueError)):
        cachepack.apply_pack(data, cachepack._sha(data))
    assert not (tmp_path / "evil.json").exists()


def test_packing_refuses_keys_outside_the_cache_and_missing_answers(cache):
    with pytest.raises(ValueError):
        cachepack.make_pack(["../../pipeline/.env"])
    with pytest.raises(cachepack.PackError, match="not in the local cache"):
        cachepack.make_pack(["llm/" + "9" * 64 + ".json"])


def test_records_round_trip_and_applied_list(cache):
    pack = cachepack.make_pack([K1, K2])
    cachepack.write_record(pack)
    r = cachepack.recorded()[pack.sha]
    assert r["keys"] == 2 and r["bytes"] == len(pack.data)
    assert cachepack.applied() == set()
    cachepack.mark_applied(pack.sha)
    assert cachepack.applied() == {pack.sha}
