"""Build the saved-reads package from recorded runs (record_all.sh): every answer with a readable image name and what
it was asked with, ready to check in.

Writes, under OUT (the checked-in folder, default pipeline/saved-reads):
  reads/<read>.jsonl.gz   one line per answer of that read, sorted by image:
                            read, series, sheet, image (readable name), page_box (page px, when the image is a piece of
                            the sheet's page.png), at (the point it is about, page px, when it is about one point),
                            case (code-check case id), variant, model, effort, prompt (short name), asked (when the
                            answer was stored, UTC), script (where the script is in git and the run's arguments),
                            answer (as the model gave it), image_sha256, cache_key
  prompts.jsonl           {name, read, sha256, text}: every prompt, by its short name
  schemas.jsonl           {sha256, schema}: every answer format
  runs.jsonl              every recorded run (script, git location, arguments, settings)
  images.manifest.tsv     readable image name, sha256, bytes
and under IMAGES (default pipeline/data/saved-reads/images, not in git): every image at its readable name; zipped by
the caller as images.zip.
usage: python -m saved_reads package --calls CALLS_DIR [--out OUT] [--images IMAGES]"""
import gzip
import json
import os
import shutil
import struct
import time
from collections import defaultdict
from pathlib import Path

from sfzmaps import config

# recorded folder name -> (read name, short description used in the README)
READS = {
    "terra-boxes": "terra-page-boxes",
    "codes2000-terra": "code-reads", "codes2000-sol": "code-reads",
    "leaders-sol": "leader-reads",
    "k8-textpass-cw": "terra-turned-tiles",
    "k8-placeread": "terra-place-reads", "k8-placeask": "terra-place-asks", "k8-placeask-split": "terra-place-asks-split",
    "k8-distask": "terra-district-asks",
    "k8-read3": "third-read", "k8-read3r": "third-reread",
    "pipeline-labels": "pipeline-label-reads",
}


def _read_of(folder: str) -> str:
    for k, v in READS.items():
        if folder == k or folder.startswith(k + "-"):
            return v
    return folder


def _png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def _find(ctx: dict, *names):
    """the first of names found in any frame's locals (innermost frames first)"""
    for frame in ctx.values():
        for n in names:
            if n in frame:
                return frame[n]
    return None


def _p4(v) -> str:
    return f"{int(round(float(v))):04d}"


def describe(read: str, script: str, g: dict, ctx: dict, wh: tuple[int, int], argv: list[str], env: dict) -> dict:
    """series, sheet, page_box / at / case / variant, and the image's readable name (without the read folder)"""
    d = {}
    w, h = wh
    name = Path(script).name
    if name == "textboxes.py":
        t = _find(ctx, "t")
        x0, y0 = g.get("bx0", 0) + t[0], g.get("by0", 0) + t[1]
        d.update(series=g["series"], sheet=g["sheet"], page_box=[x0, y0, x0 + w, y0 + h])
        d["image"] = f"{g['sheet']}/tile_{_p4(x0)}_{_p4(y0)}_{w}x{h}.png"
    elif name in ("leaders2.py", "leaders.py", "reread.py"):
        c = _find(ctx, "c") or {}
        sid = c.get("sid", "")
        sheet = sid if sid.startswith("2000-") else f"2000-digital-zoning-{sid}"
        d.update(series="2000-digital-zoning", sheet=sheet, at=[c.get("x"), c.get("y")], code_read_before=c.get("code"))
        kind = {"leaders2.py": "leader-map", "leaders.py": "leader-read", "reread.py": "recheck"}[name]
        case = f"case_{c['id']}_" if "id" in c else ""
        if "id" in c:
            d["case"] = c["id"]
        d["image"] = f"{kind}/{sheet}/{case}at_{_p4(c.get('x', 0))}_{_p4(c.get('y', 0))}.png"
    elif name == "pipeline_labels.py":
        sheet_dir = _find(ctx, "d")
        series = _find(ctx, "series") or ""
        t = _find(ctx, "t")
        sheet = Path(str(sheet_dir)).name if sheet_dir else ""
        d.update(series=series, sheet=sheet)
        if t:
            d["page_box"] = [t[0], t[1], t[0] + w, t[1] + h]
            d["image"] = f"{series}/{sheet}/tile_{_p4(t[0])}_{_p4(t[1])}_{w}x{h}.png"
    else:                                   # k8 scripts: sheet from the frame, the rest from what the frame knew
        nn = _find(ctx, "nn", "sheet_nn")
        sheet = f"2000-digital-zoning-p{nn}" if isinstance(nn, str) and len(nn) <= 3 else (_find(ctx, "sheet", "sid") or "")
        d.update(series="2000-digital-zoning", sheet=sheet)
        variant = env.get("SPLIT") and "split" or (argv[0] if name == "textpass.py" and argv else "")
        if variant:
            d["variant"] = variant
        keys = [k for k in ("place", "pi", "i", "j", "k", "tx", "ty", "x0", "y0", "x1", "y1", "x", "y", "turn", "rot")
                if _find(ctx, k) is not None and not isinstance(_find(ctx, k), (list, dict))]
        parts = [f"{k}_{_find(ctx, k)}" for k in keys[:6]]
        d["image"] = f"{sheet or 'unknown-sheet'}/" + ("_".join(parts) or "unplaced") + (f"_{variant}" if variant else "")
    return d


def main(calls: Path, out: Path, images: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "reads").mkdir(exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    cache = config.CACHE_DIR
    prompts, schemas, runs, prompt_names = {}, {}, {}, {}
    per_read = defaultdict(dict)
    manifest = {}
    for folder in sorted(p for p in calls.iterdir() if (p / "calls.jsonl").exists()):
        read = _read_of(folder.name)
        for line in open(folder / "runs.jsonl"):
            r = json.loads(line)
            runs.setdefault(r["run"], {}).update(r)
        for line in open(folder / "prompts.jsonl"):
            p = json.loads(line)
            prompts[p["sha"]] = p["text"]
            if p["sha"] not in prompt_names:
                n = 1 + sum(1 for v in prompt_names.values() if v.startswith(read + "-v"))
                prompt_names[p["sha"]] = f"{read}-v{n}"
        if (folder / "schemas.jsonl").exists():
            for line in open(folder / "schemas.jsonl"):
                s = json.loads(line)
                schemas[s["sha"]] = s["schema"]
        for line in open(folder / "calls.jsonl"):
            c = json.loads(line)
            if not c["hit"]:
                continue
            run = runs[c["run"]]
            img = folder / "images" / f"{c['image']}.png"
            d = describe(read, run["script"], run.get("script_globals", {}), c.get("context", {}), _png_size(img),
                         run.get("argv", []), run.get("env", {}))
            name = f"{read}/{d.pop('image', 'unplaced/' + c['image'][:16] + '.png')}"
            if not name.endswith(".png"):
                name += ".png"
            dest = images / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(img, dest)
            manifest[name] = (c["image"], img.stat().st_size)
            f = cache / c["key"]
            rec = {"read": read, **d, "image": name, "model": c["model"][1], "provider": c["model"][0],
                   "effort": c["effort"], "params": c["params"], "prompt": prompt_names[c["prompt"]],
                   "asked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime)),
                   "script": {"source": run.get("source", ""), "git_blob": run.get("script_git_blob"),
                              "args": run.get("argv", []), "env": run.get("env", {})},
                   "answer": json.loads(f.read_text()), "image_sha256": c["image"], "cache_key": c["key"]}
            if c.get("schema"):
                rec["answer_format"] = c["schema"]
            per_read[read][(name, c["key"])] = rec
    for read, recs in per_read.items():
        with gzip.open(out / "reads" / f"{read}.jsonl.gz", "wt", compresslevel=9) as gz:
            for k in sorted(recs):
                gz.write(json.dumps(recs[k], sort_keys=True) + "\n")
        print(f"{read}: {len(recs)} answers")
    with open(out / "prompts.jsonl", "w") as f:
        for sha, name in sorted(prompt_names.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"name": name, "sha256": sha, "text": prompts[sha]}) + "\n")
    with open(out / "schemas.jsonl", "w") as f:
        for sha, s in sorted(schemas.items()):
            f.write(json.dumps({"sha256": sha, "schema": s}, sort_keys=True) + "\n")
    with open(out / "runs.jsonl", "w") as f:
        for k in sorted(runs):
            r = {kk: v for kk, v in runs[k].items() if kk not in ("cwd", "data_dir")}
            r["script"] = r.get("source") or Path(r.get("script", "")).name
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(out / "images.manifest.tsv", "w") as f:
        f.write("image\tsha256\tbytes\n")
        for name in sorted(manifest):
            f.write(f"{name}\t{manifest[name][0]}\t{manifest[name][1]}\n")
    print(f"{sum(len(v) for v in per_read.values())} answers, {len(manifest)} images, {len(prompt_names)} prompts")
