import hashlib
import zipfile

import pytest

from sfzmaps import datapack
from sfzmaps.ship import object_path

MB = 1 << 20


def _put(cache, body: bytes, suffix: str = "") -> str:
    md5 = hashlib.md5(body).hexdigest() + suffix
    p = object_path(md5, cache)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return md5


def test_pack_roundtrip_is_deterministic(tmp_path):
    src = tmp_path / "src"
    md5s = [_put(src, b"alpha"), _put(src, b"beta"), _put(src, b'[{"md5": "x"}]', ".dir")]
    a = datapack.make_pack(md5s, src, tmp_path / "a.zip")
    b = datapack.make_pack(list(reversed(md5s)), src, tmp_path / "b.zip")
    assert a == b == hashlib.sha256((tmp_path / "a.zip").read_bytes()).hexdigest()
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()

    dst = tmp_path / "dst"
    assert datapack.apply_pack(tmp_path / "a.zip", dst) == 3
    for m in md5s:
        assert object_path(m, dst).read_bytes() == object_path(m, src).read_bytes()
    assert datapack.apply_pack(tmp_path / "a.zip", dst) == 0  # present members are skipped


def test_apply_refuses_a_member_whose_md5_is_wrong(tmp_path):
    good = hashlib.md5(b"good").hexdigest()
    liar = hashlib.md5(b"claimed").hexdigest()
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(good, b"good")
        f.writestr(liar, b"tampered")
    dst = tmp_path / "dst"
    with pytest.raises(datapack.DataPackError):
        datapack.apply_pack(z, dst)
    assert not object_path(liar, dst).exists()
    assert not object_path(good, dst).exists()  # refused before anything is written


def test_plan_splits_by_size():
    objs = {f"{i:032x}": 7 * MB for i in range(40)}  # 280 MB of small objects
    objs["f" * 32] = 9 * MB
    packs, singles = datapack.plan_packs(objs)
    assert singles == ["f" * 32]
    assert sorted(m for p in packs for m in p) == sorted(set(objs) - {"f" * 32})
    assert all(p == sorted(p) for p in packs)
    assert all(sum(objs[m] for m in p) <= datapack.PACK_MAX for p in packs)
    assert len(packs) == 2
