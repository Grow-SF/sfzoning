"""Build the saved-reads package from recorded runs (record_all.sh) and the reads matched by time (timed_reads.json):
every answer with a readable image name and what it was asked with, ready to check in.

Writes, under OUT (the checked-in folder, default pipeline/saved-reads):
  reads/<read>.jsonl.gz   one line per answer of that read, sorted by image:
                            read, series, sheet, image (readable name: <read>/<sheet>/<what>.png), page_box (page px,
                            when the image is a piece of the sheet's page), at (the point it is about, page px), case
                            (code-check case id), place (an old solve's place or district number), variant, scale,
                            model, provider, effort, prompt (short name; text in prompts.jsonl), asked (when the
                            answer was stored, UTC), script (where the script is in git, its arguments and settings),
                            matched ("replayed": the record's prompt, image and model were asked again from the cache and
                            make exactly its cache key; "by time": see timed_reads.json), answer (as the model gave it),
                            image_sha256, cache_key
  prompts.jsonl           {sha256, templates, text}: every exact prompt, and the templates (one per read) it belongs to
  schemas.jsonl           {sha256, schema}: every answer format
  runs.jsonl              every recorded run (script, git location, arguments, settings)
  images.manifest.tsv     readable image name, sha256, bytes
and under IMAGES (default pipeline/data/saved-reads/images, not in git): every image at its readable name.
usage: python -m saved_reads package --calls CALLS_DIR [--out OUT] [--images IMAGES]"""
import gzip
import json
import re
import shutil
import struct
import time
from collections import defaultdict
from pathlib import Path

from sfzmaps import config

from .by_time import stored_between

READS = {   # recorded folder -> read name
    "codes2000-terra": "code-reads", "codes2000-sol": "code-reads", "leaders-sol": "leader-reads",
    "k8-textpass-cw": "terra-turned-tiles", "k8-placeread": "terra-place-reads", "k8-placeask": "terra-place-asks",
    "k8-placeask-split": "terra-place-asks-split", "k8-distask": "terra-district-asks", "k8-read3": "third-read",
    "k8-read3r": "third-reread", "pipeline-labels": "pipeline-label-reads",
}
SER = "2000-digital-zoning"


def _read_of(folder: str) -> str:
    if folder in READS:
        return READS[folder]
    if folder.startswith("codes2000-") and folder.endswith("-fixed"):
        return "code-reads-fixed-marks"           # rerun/rerun.sh: the marks red, as the prompts say
    if folder.startswith("leaders-sol-fixed"):
        return "leader-reads-fixed-marks"
    return "terra-page-boxes" if folder.startswith("terra-boxes-") else folder


def _png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        return struct.unpack(">II", f.read(24)[16:24])


def _find(ctx: dict, name):
    for frame in ctx.values():
        if name in frame:
            return frame[name]
    return None


def _n(v) -> str:
    return f"{int(round(float(v))):04d}"


def _sheet(sid) -> str:
    sid = str(sid)
    return sid if sid.startswith(SER) else f"{SER}-p{sid.removeprefix('p')}"


def describe(script: str, g: dict, ctx: dict, note: dict, wh: tuple[int, int], argv: list[str], env: dict) -> dict:
    """series, sheet, the image's readable name (under the read's folder), and page_box / at / case / place / variant /
    scale, from what the asking script knew (record.py's context, the script's settings, the note it posted)"""
    w, h = wh
    name = Path(script).name
    if name == "textboxes.py":
        t = _find(ctx, "t")
        x0, y0 = g.get("bx0", 0) + t[0], g.get("by0", 0) + t[1]
        return {"series": g["series"], "sheet": g["sheet"], "page_box": [x0, y0, x0 + w, y0 + h],
                "image": f"{g['series']}/{g['sheet']}/tile_{_n(x0)}_{_n(y0)}_{w}x{h}.png"}
    if name in ("leaders2.py", "reread.py"):
        c = _find(ctx, "c")
        sheet = _sheet(c["sid"])
        d = {"series": SER, "sheet": sheet, "at": [c["x"], c["y"]], "code_read_before": c.get("code")}
        if name == "reread.py":
            d["case"] = c["id"]
            d["image"] = f"recheck/{sheet}/case_{c['id']}_at_{_n(c['x'])}_{_n(c['y'])}.png"
        else:
            d["image"] = f"leader-map/{sheet}/at_{_n(c['x'])}_{_n(c['y'])}.png"
        return d
    if name == "leaders.py":
        sheet, x, y = _sheet(_find(ctx, "sid")), _find(ctx, "x"), _find(ctx, "y")
        return {"series": SER, "sheet": sheet, "at": [x, y], "code_read_before": _find(ctx, "code"),
                "image": f"{sheet}/at_{_n(x)}_{_n(y)}.png"}
    if name == "pipeline_labels.py":
        t, s = _find(ctx, "t"), note.get("scale") or 1.0
        box = [round(v / s, 1) for v in (t[0], t[1], t[0] + w, t[1] + h)]
        return {"series": note["series"], "sheet": note["sheet"], "scale": s, "page_box": box,
                "image": f"{note['series']}/{note['sheet']}/tile_{_n(t[0])}_{_n(t[1])}_{w}x{h}_at_scale_{s:g}.png"}
    j = _find(ctx, "j")
    if name == "textpass.py":                       # j: sheet, x, y, w, h (page px)
        nn, x, y, tw, th = j[:5]
        var = g.get("VAR", argv[0] if argv else "")
        return {"series": SER, "sheet": _sheet(nn), "variant": var, "page_box": [x, y, x + tw, y + th],
                "image": f"{_sheet(nn)}/tile_{_n(x)}_{_n(y)}_{tw}x{th}_{var}.png"}
    if name == "placeread.py":                      # j: sheet, x, y, w, h, turned
        nn, x, y, tw, th, turned = j[:6]
        way = "turned" if turned else "upright"
        return {"series": SER, "sheet": _sheet(nn), "variant": way, "page_box": [x, y, x + tw, y + th],
                "image": f"{_sheet(nn)}/tile_{_n(x)}_{_n(y)}_{tw}x{th}_{way}.png"}
    if name in ("placeask.py", "distask.py"):       # j: place, sheet, x, y, enlarged, size (page px), turned
        place, nn, x, y, up, size, turned = j[:7]
        way = "turned" if turned else "upright"
        what = "district" if name == "distask.py" else "place"
        return {"series": SER, "sheet": _sheet(nn), "place": place, "at": [x, y], "variant": way,
                "image": f"{_sheet(nn)}/{what}_{place}_at_{_n(x)}_{_n(y)}_{size}px_x{up:g}_{way}.png"}
    return {"image": None}


class Templates:
    """Prompts that differ only in their numbers (tile size, place) or in the code asked about are one template:
    the prompt with the code it asks about (quoted on the line that says it is circled) replaced by "{code}"
    and every number by #.  Named <read>-t1, -t2 ... in order of first use."""

    def __init__(self):
        self.names, self.count = {}, defaultdict(int)

    def name(self, read: str, text: str, code: str | None) -> str:
        t = text
        if code:                    # the code varies only on the line that says it is circled (at the centre)
            t = "\n".join(l.replace(f'"{code}"', '"{code}"') if "circled" in l else l for l in text.split("\n"))
        t = re.sub(r"\d+(\.\d+)?", "#", t)
        k = (read, t)
        if k not in self.names:
            self.names[k] = f"{read}-t{1 + sum(r == read for r, _ in self.names)}"
        self.count[self.names[k]] += 1
        return self.names[k]


def _arg(a: str) -> str:
    """a script argument as a reader can use it: no machine-specific folders"""
    pipe = str(config.DATA_DIR.parent)
    if a.startswith(pipe + "/data/saved-reads/work"):
        return "<the replay's output folder>"
    if a.startswith(pipe + "/"):
        return a[len(pipe) + 1:]
    if "/" in a and Path(a).is_absolute():
        return Path(a).name
    return a


def _asked(key: str) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((config.CACHE_DIR / key).stat().st_mtime))


def main(calls: Path, out: Path, images: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "reads").mkdir(exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    prompts, schemas, runs, prompt_read, used = {}, {}, {}, {}, defaultdict(set)
    templates = Templates()
    per_read, manifest, unplaced = defaultdict(dict), {}, defaultdict(int)
    for folder in sorted(p for p in calls.iterdir() if (p / "calls.jsonl").exists()):
        read = _read_of(folder.name)
        for line in open(folder / "runs.jsonl"):
            r = json.loads(line)
            runs.setdefault(r["run"], {}).update(r)
        for line in open(folder / "prompts.jsonl"):
            p = json.loads(line)
            prompts[p["sha"]] = p["text"]
            prompt_read.setdefault(p["sha"], read)
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
            d = describe(run["script"], run.get("script_globals", {}), c.get("context", {}), c.get("note", {}),
                         _png_size(img), run.get("argv", []), run.get("env", {}))
            rel = d.pop("image")
            if rel is None:
                unplaced[read] += 1
                rel = f"unplaced/{c['image'][:16]}.png"
            name = f"{read}/{rel}"
            if name in manifest and manifest[name][0] != c["image"]:
                raise SystemExit(f"two different images named {name}")
            dest = images / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(img, dest)
            manifest[name] = (c["image"], img.stat().st_size)
            tname = templates.name(read, prompts[c["prompt"]], d.get("code_read_before"))
            used[c["prompt"]].add(tname)             # a prompt asked by two reads belongs to both templates
            per_read[read][(name, c["key"])] = {
                "read": read, **d, "image": name, "model": c["model"][1], "provider": c["model"][0],
                "effort": c["effort"], "params": c["params"], "prompt": tname, "prompt_sha256": c["prompt"],
                "answer_format": c.get("schema"), "asked": _asked(c["key"]),
                "script": {"source": run.get("source", ""), "git_blob": run.get("script_git_blob"),
                           "args": [_arg(a) for a in run.get("argv", [])],
                           "env": {k: _arg(v) for k, v in run.get("env", {}).items()}},
                "matched": "replayed", "answer": json.loads((config.CACHE_DIR / c["key"]).read_text()),
                "image_sha256": c["image"], "cache_key": c["key"]}
    done = {k for recs in per_read.values() for (_, k) in recs}
    spec = json.loads((Path(__file__).parent / "timed_reads.json").read_text())
    for t in spec["reads"]:
        n = 0
        for m, key, answer in stored_between(t["from"], t["to"]):
            if key in done or (t["fields"] and not set(t["fields"]) <= set(answer)):
                continue
            done.add(key)
            n += 1
            per_read[t["read"]][("~" + key, key)] = {
                "read": t["read"], "series": SER, "sheet": None, "image": None, "model": t["model"],
                "asked": _asked(key), "matched": "by time",
                "script": {"source": f"rescued scratch copy: pipeline/data/rnd/scratch-2026-11-13/{t['script']}",
                           "args": t["args"], "env": t.get("env", {}), "command": t["command"],
                           "transcript_line": t["line"], "window": [t["from"], t["to"]]},
                "answer": answer, "cache_key": key}
        print(f"  by time, {t['read']}: {n} answers")
    for read, recs in sorted(per_read.items()):
        with gzip.open(out / "reads" / f"{read}.jsonl.gz", "wt", compresslevel=9) as gz:
            for k in sorted(recs):
                gz.write(json.dumps(recs[k], sort_keys=True) + "\n")
        by = defaultdict(int)
        for r in recs.values():
            by[r["matched"]] += 1
        print(f"{read}: {len(recs)} answers ({dict(by)}){f'; {unplaced[read]} without a place' if unplaced[read] else ''}")
    with open(out / "prompts.jsonl", "w") as f:            # every exact prompt, with the templates it belongs to
        for sha, tnames in sorted(used.items(), key=lambda kv: (sorted(kv[1]), kv[0])):
            f.write(json.dumps({"sha256": sha, "templates": sorted(tnames), "text": prompts[sha]}) + "\n")
    with open(out / "templates.jsonl", "w") as f:          # each template: the prompt with what varies blanked out
        for (read, text), tname in sorted(templates.names.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"template": tname, "read": read, "prompts": templates.count[tname], "text": text}) + "\n")
    with open(out / "schemas.jsonl", "w") as f:
        for sha, s in sorted(schemas.items()):
            f.write(json.dumps({"sha256": sha, "schema": s}, sort_keys=True) + "\n")
    with open(out / "runs.jsonl", "w") as f:
        for k in sorted(runs):
            r = {kk: v for kk, v in runs[k].items() if kk not in ("cwd", "data_dir", "script")}
            r["argv"] = [_arg(a) for a in r.get("argv", [])]
            r["env"] = {kk: _arg(v) for kk, v in r.get("env", {}).items()}
            if "script_globals" in r:
                r["script_globals"] = {kk: _arg(v) if isinstance(v, str) else v for kk, v in r["script_globals"].items()}
            r["script"] = r.pop("source", "") or Path(runs[k].get("script", "")).name
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(out / "images.manifest.tsv", "w") as f:
        f.write("image\tsha256\tbytes\n")
        for name in sorted(manifest):
            f.write(f"{name}\t{manifest[name][0]}\t{manifest[name][1]}\n")
    print(f"{sum(len(v) for v in per_read.values())} answers, {len(manifest)} images, {len(used)} prompts in "
          f"{len(templates.names)} templates")
