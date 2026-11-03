"""Delete from the Blob store what no shipped commit still uses.

    uv run python -m sfzmaps.prune --dry-run      # says what it would delete; deletes nothing
    uv run python -m sfzmaps.prune                # deletes it

(It isn't an `sfzmaps` subcommand yet: cli.py is a dependency of every build stage, so adding one
means re-recording dvc.lock and shipping again.)

*Used* is everything named by `main`'s shipped manifest, and by that of every open pull request
whose branch has a pipeline/cache-needed.json: the manifest itself, the needed-answers list, every
answer pack (zip and index), every data pack and every object stored alone. Everything else under
`dvc/` and `cache/` is unused. Tiles and stats under other prefixes are never listed or touched.

It refuses when main's manifest isn't version 2 (a v1 manifest stores every object alone, and
pruning would delete objects older commits still pull), when an open pull request's branch can't
be read, and when any used name is missing from the store's listing (something is already wrong;
deleting more would not help).

Requests: one listing pass (one call per 1,000 names), the few manifests not already cached
locally, then (unless --dry-run) site/scripts/blob-del.mjs deletes in batches of 100 URLs per
del() call, at most 50 deletes a second (Vercel's Pro limit is 75 advanced operations a second,
counted per blob; deletes are free), and one more listing confirms.

Commits shipped before this was run can't be pulled afterwards if their objects were pruned.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field

from sfzmaps import config, ship

PREFIXES = ("dvc/", "cache/")
BATCH = 100                 # URLs per del() call (blob-del.mjs)
DELETES_PER_S = 50          # pacing in blob-del.mjs; Vercel Pro allows 75 advanced operations/s

KIND_PATTERNS = (
    ("dvc directory manifests", re.compile(r"^dvc/files/md5/[0-9a-f]{2}/[0-9a-f]{30}\.dir$")),
    ("dvc single objects", re.compile(r"^dvc/files/md5/[0-9a-f]{2}/[0-9a-f]{30}$")),
    ("data packs", re.compile(r"^cache/datapacks/[0-9a-f]{64}\.zip$")),
    ("answer packs", re.compile(r"^cache/packs/[0-9a-f]{64}\.zip$")),
    ("answer pack indexes", re.compile(r"^cache/packs/[0-9a-f]{64}\.keys\.gz$")),
    ("needed lists", re.compile(r"^cache/needed/[0-9a-f]{64}\.keys\.gz$")),
    ("shipped manifests", re.compile(r"^cache/shipped/[0-9a-f]{64}\.json$")),
)
UNRECOGNIZED = "unrecognized"


class PruneError(Exception):
    pass


def kind(name: str) -> str:
    for k, pat in KIND_PATTERNS:
        if pat.match(name):
            return k
    return UNRECOGNIZED


def used_names(needed: dict, manifest: dict, records=()) -> set[str]:
    """Every name one shipped commit uses: its manifest, needed list, answer packs (the
    manifest's, plus the pack records at that commit), data packs and objects stored alone."""
    out = {f"{ship.SHIPPED_PREFIX}/{needed['shipped_sha']}.json",
           f"{ship.NEEDED_PREFIX}/{needed['keys_sha']}.keys.gz"}
    for sha in {*(manifest.get("packs") or {}), *records}:
        out |= {f"{ship.PACK_PREFIX}/{sha}.zip", f"{ship.PACK_PREFIX}/{sha}.keys.gz"}
    where = (manifest.get("where") or {}) if manifest.get("version") == 2 else {}
    for sha in {*(manifest.get("data_packs") or {}), *where.values()}:
        out.add(ship.datapack_name(sha))
    out |= {ship.object_name(m) for m in manifest.get("objects") or {} if m not in where}
    return out


@dataclass
class Plan:
    delete: dict[str, int] = field(default_factory=dict)
    keep: dict[str, int] = field(default_factory=dict)
    unrecognized: dict[str, int] = field(default_factory=dict)   # kept, reported
    missing: list[str] = field(default_factory=list)              # used but not on the store


def plan(held: dict[str, int], used: set[str]) -> Plan:
    """Split a listing (name -> bytes) into what to keep and what to delete. Only names in the
    content-addressed layouts ship writes are ever deleted."""
    p = Plan(missing=sorted(used - set(held)))
    for name, size in held.items():
        if not name.startswith(PREFIXES):
            continue
        if name in used:
            p.keep[name] = size
        elif kind(name) == UNRECOGNIZED:
            p.unrecognized[name] = size
        else:
            p.delete[name] = size
    return p


def by_kind(names: dict[str, int]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for n, s in names.items():
        c, b = out.get(kind(n), (0, 0))
        out[kind(n)] = (c + 1, b + s)
    return dict(sorted(out.items()))


# --- what's in use ---------------------------------------------------------------------------------

def _show(ref: str, path: str) -> str | None:
    try:
        return ship._git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return None


def open_pr_refs() -> list[str]:
    p = subprocess.run(["gh", "pr", "list", "--state", "open", "--limit", "200",
                        "--json", "number,headRefName,isCrossRepository"],
                       cwd=ship._repo_root(), capture_output=True, text=True)
    if p.returncode != 0:
        raise PruneError(f"`gh pr list` failed: {p.stderr.strip()[-300:]}")
    refs = []
    for pr in json.loads(p.stdout or "[]"):
        if pr.get("isCrossRepository"):
            raise PruneError(f"PR #{pr['number']} comes from a fork; its cache-needed.json can't be "
                             "read from origin")
        refs.append(f"origin/{pr['headRefName']}")
    return refs


def gather_used(client=None) -> tuple[set[str], list[str], int]:
    """(used names, the refs they came from, manifests downloaded). Fetches origin first."""
    ship._git("fetch", "-q", "origin")
    client = client or ship._client()
    downloads = 0
    used: set[str] = set()
    sources = []
    for i, ref in enumerate(["origin/main", *open_pr_refs()]):
        try:
            ship._git("rev-parse", "--verify", "-q", ref)
        except subprocess.CalledProcessError:
            raise PruneError(f"{ref} isn't on origin; can't tell what it uses") from None
        text = _show(ref, "pipeline/cache-needed.json")
        if text is None:
            if i == 0:
                raise PruneError("origin/main has no pipeline/cache-needed.json")
            continue
        needed = json.loads(text)
        if not needed.get("shipped_sha"):
            raise PruneError(f"{ref}'s cache-needed.json names no shipped manifest")
        cached = (config.CACHE_DIR / "shipped" / f"{needed['shipped_sha']}.json").exists()
        manifest = ship._manifest(needed["shipped_sha"], client)
        downloads += not cached
        if i == 0 and manifest.get("version") != 2:
            raise PruneError(f"main's shipped manifest is version {manifest.get('version', 1)}, not 2: "
                             "pruning would delete the objects it stores alone")
        used |= used_names(needed, manifest, ship.records_at(ref))
        sources.append(f"{ref} (manifest v{manifest.get('version', 1)})")
    return used, sources, downloads


# --- deleting --------------------------------------------------------------------------------------

def delete(names: list[str]) -> None:
    """site/scripts/blob-del.mjs: batches of BATCH URLs, at most DELETES_PER_S a second."""
    entries = [{"name": n, "url": f"{ship.BASE}/{n}"} for n in names]
    p = ship._node("blob-del.mjs", [], json.dumps(entries))
    if p.returncode != 0:
        raise PruneError(f"deleting stopped: {p.stderr.strip()[-300:]} (run prune again; it lists first)")


def run(dry_run: bool, echo=print) -> int:
    used, sources, downloads = gather_used()
    held = ship.listing(PREFIXES)
    list_calls = sum(max(1, math.ceil(sum(n.startswith(pre) for n in held) / 1000)) for pre in PREFIXES)
    p = plan(held, used)
    echo("used by: " + ", ".join(sources))
    if p.missing:
        raise PruneError(f"{len(p.missing)} used name(s) aren't on the store (e.g. {p.missing[0]}); "
                         "not deleting anything")
    gb = lambda b: f"{b / 1e9:.2f} GB"
    for title, names in (("delete", p.delete), ("keep", p.keep)):
        echo(f"{title}: {len(names)} name(s), {gb(sum(names.values()))}")
        for k, (c, b) in by_kind(names).items():
            echo(f"  {k}: {c}, {gb(b)}")
    if p.unrecognized:
        echo(f"unrecognized (kept): {len(p.unrecognized)}, e.g. {sorted(p.unrecognized)[0]}")
    batches = math.ceil(len(p.delete) / BATCH)
    echo(f"requests so far: {list_calls} listing call(s), {downloads} manifest download(s)")
    if dry_run:
        echo(f"a real run would make {list_calls + batches + list_calls} request(s): {list_calls} "
             f"listing, {batches} delete call(s) of up to {BATCH}, {list_calls} listing to confirm; "
             f"deleting takes about {math.ceil(len(p.delete) / DELETES_PER_S)} s")
        return 0
    if p.delete:
        delete(sorted(p.delete))
    after = ship.listing(PREFIXES)
    left = sorted(set(p.delete) & set(after))
    gone = [n for n in used if n not in after]
    echo(f"deleted {len(p.delete) - len(left)} name(s), {gb(sum(s for n, s in p.delete.items() if n not in after))}")
    if left or gone:
        raise PruneError(f"after deleting: {len(left)} unused name(s) remain, {len(gone)} used name(s) "
                         "are missing")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sfzmaps.prune", description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", "-n", action="store_true", help="Say what would be deleted; delete nothing.")
    args = ap.parse_args(argv)
    try:
        return run(args.dry_run)
    except (PruneError, ship.ShipError) as e:
        print(f"prune: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
