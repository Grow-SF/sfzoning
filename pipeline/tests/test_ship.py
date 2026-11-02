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
        self.refuse = {}             # name -> how many times to answer 403 first

    def client(self):
        def handle(req):
            self.requests.append(req)
            name = req.url.path.lstrip("/")
            if name in self.refuse:
                self.refuse[name] -= 1
                if self.refuse[name] >= 0:
                    return httpx.Response(403)
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


class Store:
    """The Blob store as ship sees it: a public FakeBlob plus the token-side list / put."""
    def __init__(self, blob):
        self.blob, self.lists, self.puts = blob, 0, []

    def listing(self, prefixes=None):
        self.lists += 1
        return {n: len(b) for n, b in self.blob.objects.items()}

    def upload(self, entries):
        for e in entries:
            self.puts.append(e["name"])
            self.blob.objects[e["name"]] = open(e["file"], "rb").read()
        return {e["name"]: "uploaded" for e in entries}


F1 = b"tile bytes"


def _setup_ship(data, tmp_path, monkeypatch, blob):
    manifest = json.dumps([{"md5": md5(F1), "relpath": "x.pmtiles"}]).encode()
    dirmd5 = md5(manifest) + ".dir"
    cache = _cache_with(tmp_path, {dirmd5: manifest, md5(F1): F1})
    config.PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    (config.PIPELINE_DIR / "dvc.lock").write_text(f"stages:\n  s:\n    outs:\n    - path: data/out/x\n      md5: {dirmd5}\n")
    for k in (K1, K2):
        p = config.CACHE_DIR / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    store = Store(blob)
    monkeypatch.setattr(ship, "preflight", lambda base="origin/main": None)
    monkeypatch.setattr(ship, "dvc_cache", lambda: cache)
    monkeypatch.setattr(ship, "_client", blob.client)
    monkeypatch.setattr(ship, "needed_answers", lambda: ({K1, K2}, []))
    monkeypatch.setattr(ship, "records_at", lambda ref: {})
    monkeypatch.setattr(ship, "known_keys", lambda recs, client=None: {K1: "x"})   # main has K1
    commits = []
    monkeypatch.setattr(ship, "_git", lambda *a: commits.append(a) or "")
    monkeypatch.setattr(ship, "listing", store.listing)
    monkeypatch.setattr(ship, "upload", store.upload)
    return cache, commits, dirmd5, store



def test_ship_refuses_to_overwrite_an_object_that_differs(data, tmp_path, monkeypatch):
    blob = FakeBlob()
    _setup_ship(data, tmp_path, monkeypatch, blob)
    blob.objects[ship.object_name(md5(F1))] = b"other bytes!!"
    with pytest.raises(ship.ShipError, match="differ in size"):
        ship.ship()




def test_ship_refuses_artifacts_of_unknown_provenance(data, monkeypatch):
    monkeypatch.setattr(ship, "preflight", lambda base="origin/main": None)
    monkeypatch.setattr(ship, "needed_answers", lambda: (set(), [data / "labels.json"]))
    monkeypatch.setattr(ship, "_client", FakeBlob().client)
    with pytest.raises(ship.ShipError, match="no answers record"):
        ship.ship()



# --- the merge gate ---------------------------------------------------------------------------------

def test_lock_problems_flag_code_changes_and_ignore_missing_data():
    status = json.dumps({
        "edition-1948": [{"changed deps": {"src/sfzmaps/runner.py": "modified", "data/raw": "deleted"}},
                         {"changed outs": {"data/out/1948": "deleted"}}],
        "today": ["changed command"],
        "shared": [{"changed deps": {"data/out/today/districts": "deleted"}}],
    })
    got = ship.lock_problems(status)
    assert any("src/sfzmaps/runner.py" in p for p in got) and any("today" in p for p in got)
    assert not any("data/" in p for p in got) and len(got) == 2
    assert ship.lock_problems("{}") == []




def test_no_needed_list_means_not_shipped(data, tmp_path):
    config.PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    assert any("cache-needed.json is missing" in p for p in ship.check_shipped("{}", FakeBlob().client()))
