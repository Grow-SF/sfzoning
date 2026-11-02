import hashlib
import json

import httpx
import pytest

from sfzmaps import answers, cachepack, config, ship

K1 = "llm/" + "1" * 64 + ".json"
K2 = "llm/tiles/" + "2" * 64 + ".json"


class FakeBlob:
    """Public reads of a Blob store: name -> bytes. Records every request."""
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.requests = []

    def client(self):
        def handle(req):
            self.requests.append(req)
            assert "check" in req.url.params                 # never a cacheable real-URL miss
            name = req.url.path.lstrip("/")
            if name not in self.objects:
                return httpx.Response(404)
            body = self.objects[name]
            return httpx.Response(200, headers={"content-length": str(len(body))},
                                  content=b"" if req.method == "HEAD" else body)
        return httpx.Client(transport=httpx.MockTransport(handle), base_url=ship.BASE)


def md5(b):
    return hashlib.md5(b).hexdigest()


def test_lock_outs_and_objects_list_manifests_and_their_files():
    lock = "stages:\n  a:\n    outs:\n    - path: data/out/x\n      md5: abc.dir\n    - path: f\n      md5: def\n"
    outs = ship.lock_outs(lock)
    assert outs == ["abc.dir", "def"]
    assert ship.objects(outs, lambda m: [{"md5": "f1", "relpath": "a"}, {"md5": "f2", "relpath": "b"}]) == \
        ["abc.dir", "def", "f1", "f2"]
    assert ship.object_name("0123456789abcdef0123456789abcdef") == "dvc/files/md5/01/23456789abcdef0123456789abcdef"


def test_download_checks_the_hash_and_writes_nothing_on_a_mismatch(tmp_path):
    good = b"hello"
    blob = FakeBlob({"dvc/files/md5/aa/b": good})
    dest = tmp_path / "o"
    with pytest.raises(ship.ShipError, match="don't match"):
        ship.download("dvc/files/md5/aa/b", dest, "md5", "0" * 32, blob.client())
    assert not dest.exists() and not list(tmp_path.iterdir())
    ship.download("dvc/files/md5/aa/b", dest, "md5", md5(good), blob.client())
    assert dest.read_bytes() == good
    with pytest.raises(ship.ShipError, match="not on the Blob store"):
        ship.download("dvc/files/md5/aa/c", tmp_path / "p", "md5", md5(good), blob.client())


@pytest.fixture
def data(tmp_path, monkeypatch):
    for n in ("DATA_DIR", "WORK_DIR", "OUT_DIR", "RAW_DIR", "CACHE_DIR"):
        monkeypatch.setattr(config, n, tmp_path / "data" / n.lower().removesuffix("_dir"))
    monkeypatch.setattr(config, "PIPELINE_DIR", tmp_path / "pipeline")
    return tmp_path


def _page(work, series, sid, kind):
    d = work / series / sid
    d.mkdir(parents=True)
    (d / "page.json").write_text(json.dumps({"kind": kind, "sheet_number": None, "neighbor_refs": [],
                                             "mask_boxes": []}))
    return d


def test_needed_answers_come_from_live_artifacts_only(data, monkeypatch):
    from sfzmaps import pages
    from sfzmaps.pages import Page
    work = config.WORK_DIR
    live = _page(work, "1964", "p1", "map")
    text = _page(work, "1964", "p2", "text")            # reclassified: its old labels are stale
    gone = _page(work, "1964", "p3", "map")             # no longer a discovered page
    for d, k in ((live, K1), (text, K2), (gone, K2)):
        answers.write(d / "page.json", [])
        (d / "labels.json").write_text("[]")
        answers.write(d / "labels.json", [k])
    monkeypatch.setattr(pages, "discover", lambda raw: [Page("1964", "p1", data / "a.pdf", 0),
                                                        Page("1964", "p2", data / "a.pdf", 1)])
    keys, unknown = ship.needed_answers()
    assert keys == {K1} and unknown == []
    answers.sidecar(live / "labels.json").unlink()      # provenance unknown, no baseline
    keys, unknown = ship.needed_answers()
    assert unknown == [live / "labels.json"]


def test_known_keys_refuses_packs_that_disagree(data, monkeypatch):
    import gzip
    def idx(k, h):
        return gzip.compress(f"{k} {h}\n".encode(), mtime=0)
    a, b = idx(K1, "a" * 64), idx(K1, "b" * 64)
    blob = FakeBlob({f"cache/packs/{'1' * 64}.keys.gz": a, f"cache/packs/{'2' * 64}.keys.gz": b})
    recs = {"1" * 64: {"index_sha": hashlib.sha256(a).hexdigest()},
            "2" * 64: {"index_sha": hashlib.sha256(b).hexdigest()}}
    assert ship.known_keys({"1" * 64: recs["1" * 64]}, blob.client()) == {K1: "a" * 64}
    with pytest.raises(ship.ShipError, match="disagree"):
        ship.known_keys(recs, blob.client())


def _cache_with(tmp_path, files):
    cache = tmp_path / "dvccache"
    for m, body in files.items():
        p = ship.object_path(m, cache)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    return cache


def _setup_ship(data, tmp_path, monkeypatch, blob, uploaded):
    f1 = b"tile bytes"
    manifest = json.dumps([{"md5": md5(f1), "relpath": "x.pmtiles"}]).encode()
    dirmd5 = md5(manifest) + ".dir"
    cache = _cache_with(tmp_path, {dirmd5: manifest, md5(f1): f1})
    (config.PIPELINE_DIR).mkdir(parents=True, exist_ok=True)
    (config.PIPELINE_DIR / "dvc.lock").write_text(f"stages:\n  s:\n    outs:\n    - path: data/out/x\n      md5: {dirmd5}\n")
    for k in (K1, K2):
        p = config.CACHE_DIR / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    monkeypatch.setattr(ship, "preflight", lambda base="origin/main": None)
    monkeypatch.setattr(ship, "dvc_cache", lambda: cache)
    monkeypatch.setattr(ship, "_client", blob.client)
    monkeypatch.setattr(ship, "needed_answers", lambda: ({K1, K2}, []))
    monkeypatch.setattr(ship, "records_at", lambda ref: {})
    monkeypatch.setattr(ship, "known_keys", lambda recs, client=None: {K1: "x"})   # main has K1
    commits = []
    monkeypatch.setattr(ship, "_git", lambda *a: commits.append(a) or "")
    monkeypatch.setattr(ship, "upload", lambda entries: uploaded.extend(entries) or {e["name"]: "uploaded" for e in entries})
    return cache, commits, dirmd5, md5(f1)


def test_ship_uploads_only_what_main_lacks_then_records_the_pack(data, tmp_path, monkeypatch):
    blob, uploaded = FakeBlob(), []
    cache, commits, dirmd5, filemd5 = _setup_ship(data, tmp_path, monkeypatch, blob, uploaded)
    rep = ship.ship()
    names = [e["name"] for e in uploaded]
    assert ship.object_name(dirmd5) in names and ship.object_name(filemd5) in names
    pack_zip = [n for n in names if n.endswith(".zip")]
    assert len(pack_zip) == 1 and rep.new_answers == 1                # only K2: main has K1
    rec = cachepack.recorded()[rep.pack]
    assert rec["keys"] == 1 and any(a[0] == "commit" for a in commits)
    needed = json.loads(ship.needed_path().read_text())          # the full list, for CI
    assert needed["keys"] == 2 and needed["lock_md5"] == ship.lock_md5()
    assert f"cache/needed/{needed['keys_sha']}.keys.gz" in names


def test_a_failed_upload_records_nothing(data, tmp_path, monkeypatch):
    blob, uploaded = FakeBlob(), []
    _setup_ship(data, tmp_path, monkeypatch, blob, uploaded)
    def fail(entries):
        raise ship.ShipError("upload failed")
    monkeypatch.setattr(ship, "upload", fail)
    with pytest.raises(ship.ShipError):
        ship.ship()
    assert cachepack.recorded() == {} and not ship.needed_path().exists()


def test_ship_refuses_artifacts_of_unknown_provenance(data, monkeypatch):
    monkeypatch.setattr(ship, "preflight", lambda base="origin/main": None)
    monkeypatch.setattr(ship, "needed_answers", lambda: (set(), [data / "labels.json"]))
    monkeypatch.setattr(ship, "_client", FakeBlob().client)
    with pytest.raises(ship.ShipError, match="no answers record"):
        ship.ship()


def test_pull_applies_packs_and_fetches_missing_objects_once(data, tmp_path, monkeypatch):
    # a shipped state: one pack and one output (manifest + file) on the store
    for k in (K1,):
        p = config.CACHE_DIR / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"a": 1}')
    pack = cachepack.make_pack([K1])
    cachepack.write_record(pack)
    (config.CACHE_DIR / K1).unlink()
    f1 = b"tile bytes"
    manifest = json.dumps([{"md5": md5(f1), "relpath": "x.pmtiles"}]).encode()
    dirmd5 = md5(manifest) + ".dir"
    (config.PIPELINE_DIR / "dvc.lock").write_text(f"stages:\n  s:\n    outs:\n    - path: data/out/x\n      md5: {dirmd5}\n")
    blob = FakeBlob({f"cache/packs/{pack.sha}.zip": pack.data, ship.object_name(dirmd5): manifest,
                     ship.object_name(md5(f1)): f1})
    cache = tmp_path / "dvccache"
    monkeypatch.setattr(ship, "dvc_cache", lambda: cache)
    monkeypatch.setattr(ship, "_client", blob.client)
    assert ship.pull(dvc_checkout=False) == (1, 2)
    assert (config.CACHE_DIR / K1).read_text() == '{"a": 1}'
    assert ship.object_path(md5(f1), cache).read_bytes() == f1
    assert ship.pull(dvc_checkout=False) == (0, 0)             # applied twice: a no-op
