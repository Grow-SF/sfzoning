"""Check a package against itself, without the answer cache or a replay: for every "replayed" record, rebuild the cache
key from the record's own parts exactly as sfzmaps.llm does (provider, model, prompt text, answer schema, call
parameters, and the image's PNG bytes) and compare it with the record's cache_key; and check every image's sha256
against images.manifest.tsv.  A match means the record's prompt, model and image are the ones its answer was given for.
usage: python -m saved_reads verify [--from PACKAGE] [--images IMAGES]"""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def key(provider: str, model: str, prompt: str, schema: dict, params: list[str], png: bytes) -> str:
    h = hashlib.sha256()
    for part in [provider, model, prompt, json.dumps(schema, sort_keys=True), *params]:
        h.update(part.encode())
        h.update(b"\0")
    h.update(png)
    return h.hexdigest()


def main(package: Path, images: Path) -> None:
    prompts = {json.loads(l)["sha256"]: json.loads(l)["text"] for l in open(package / "prompts.jsonl")}
    schemas = {json.loads(l)["sha256"]: json.loads(l)["schema"] for l in open(package / "schemas.jsonl")}
    manifest = {}
    for line in list(open(package / "images.manifest.tsv"))[1:]:
        name, sha, _ = line.rstrip("\n").split("\t")
        manifest[name] = sha
    n = Counter()
    bad = []
    for f in sorted((package / "reads").glob("*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if r["matched"] != "replayed":
                n["by time (no image to check)"] += 1
                continue
            img = images / r["image"]
            if not img.exists():
                n["image missing"] += 1
                continue
            png = img.read_bytes()
            if hashlib.sha256(png).hexdigest() != manifest.get(r["image"]) or manifest[r["image"]] != r["image_sha256"]:
                bad.append(("image sha256", r["image"]))
                continue
            k = key(r["provider"], r["model"], prompts[r["prompt_sha256"]], schemas[r["answer_format"]], r["params"], png)
            if f"llm/{k}.json" == r["cache_key"]:
                n["key rebuilt exactly"] += 1
            else:
                bad.append(("key", r["image"]))
    print(dict(n))
    print(f"mismatches: {len(bad)}", bad[:5])
