"""The frame finder's run (issue #71, spec parts 6-8 and the amendment "removed blocks"): held-out folds x seeds, the
step count, the temperature, the scores, the final models and every map page's outputs.  It orchestrates only: the
network is `model`, training `learn`, the cut `decide`, the scores `scores` and the temperature `calibrate`.

`run_folds` scores every labelled sheet with networks that never saw it.  Each of the `folds.OUTER_K` folds trains one
net per seed in `SEEDS` on its training sheets, logging the held-out curve and saving a snapshot at every check
(`snapshot_path`).  The step count is then chosen once, from the curves pooled over every fold and seed
(`choose_step`), and the held sheets go through two passes at that step's snapshots:

1. the logits of each seed, from which calibration px are drawn (`calibrate.sample`, one generator for the whole run,
   sheets in sorted order); the one temperature T is fitted on every held sheet together, with its sheet-bootstrap
   interval and the reliability table;
2. the logits again, the chances at T (the seeds' tempered softmaxes averaged, `model.combine`), the decision at the
   owner's cost ratio (`decide.COST_FRAME`) and the outputs, and the scores at that cut and at each of
   `RATIOS_SHOWN`.  Seed disagreement is the ink share where each seed alone, at T, decides differently.

The logits are recomputed in pass 2, not stored: a forward pass is about 50 ms a page, while two seeds' float32
logits of every held sheet would be gigabytes.  For the same reason no page ever holds every net's logits at once
when it does not need them apart: the chances are summed net by net.

The baseline is the old two-pass model's held-out `map_prob.npz` under `BASELINE` (key `map`, P(inside)), read as the
chances (map, 1 - map, 0), decided and scored the same way with no temperature.  Only `frame_kept`, `map_cut` and
`cost` are compared (paired, new - old, over the sheets both have): the old model has no removed-block class, so its
log-loss is not comparable.

`train_final` trains `SEEDS` on every labelled sheet for the chosen step count; `predict_all` predicts every other
map page with them, averaged at T; the labelled sheets keep their held-out outputs.

Outputs, all under `out` (default `data/frame`):
- `<series>/<sheet>/outside.png` (`OUTSIDE_PNG`): 1-bit, the deletion (`Decision.deleted`: outside the frame or a
  removed block);
- `<series>/<sheet>/outside_prob.npz` (`PROB_FILE`): key `q`, the chance of deletion p1 + p2 as uint8 levels
  round(255 q), for reading and placement to weigh ink by.  The cut is decided on the float chances before rounding,
  so the rounding never changes it; uint8 compresses in a third of float16's time (ruling R6);
- `<series>/<sheet>/erased.png` (`ERASED_PNG`): the scan in its own mode (1-bit as 8-bit grey) with the deleted px
  painted plain paper, the per-channel median of the kept px: for looking only;
- `<series>/<sheet>/outline.json` (`OUTLINE_FILE`): {"polygons": the kept region (`decide.outline`), "maps": n};
- `folds.json` (`FOLDS_FILE`): the held-out run (`run_folds`), deleted when a run starts, so a run that stops part
  way leaves no summary for the final training to trust;
- `folds/fold-<k>/seed-<s>/step-<n>.pt`: the folds' snapshots;
- `model/seed-<s>.pt` and `model/model.json` (`MODEL_DIR`): the final nets, their step count and T;
- `pages.json` (`PAGES_FILE`): every map page's filled and dropped ink, most dropped first, the pages to look at.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from sfzmaps.frame import calibrate, data, decide, folds, learn, model, scores
from sfzmaps.frame.data import DATA_DIR, OUT, SheetId

log = logging.getLogger(__name__)

SEEDS = (0, 1)                      # spec part 6: 2 seeds per fold, and the final ensemble's size
RATIOS_SHOWN = (2.0, 3.0, 10.0)     # cost ratios (frame : map) reported to the owner; the cut uses decide.COST_FRAME
BASELINE = DATA_DIR / "frame-cmp-none"   # the two-pass model's held-out outputs (spec part 7)
SAVE_THREADS = 4                    # computational: pages saved at once while the next is predicted

FOLDS_FILE = "folds.json"
MODEL_DIR = "model"
OUTSIDE_PNG = "outside.png"
PROB_FILE = "outside_prob.npz"
ERASED_PNG = "erased.png"
OUTLINE_FILE = "outline.json"
PAGES_FILE = "pages.json"
BASELINE_FILE = "map_prob.npz"      # the two-pass run's file name
COMPARED = ("frame_kept", "map_cut", "cost")   # the scores compared with the baseline


def sheet_dir(out: Path, sid: SheetId) -> Path:
    return out / sid[0] / sid[1]


def snapshot_path(out: Path, fold: int, seed: int, step: int) -> Path:
    """Where fold `fold`'s seed `seed` saved its net at `step` (ruling R1)."""
    return out / "folds" / f"fold-{fold}" / f"seed-{seed}" / f"step-{step:05d}.pt"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _lum(sid: SheetId) -> np.ndarray:
    return data.load_luminance(data.page_path(*sid, work=data.WORK))


def labelled_pages(labelled: Sequence[SheetId]) -> dict[SheetId, learn.Page]:
    """Each labelled sheet's luminance and class mask (0 map, 1 outside the frame, 2 removed block)."""
    pages = {}
    for sid in labelled:
        lum = _lum(sid)
        cls = data.class_mask(data.load_label(*sid, root=data.LABELS), data.load_remove(*sid, root=data.LABELS),
                              lum.shape)
        pages[sid] = learn.Page(sid, lum, cls)
    return pages


def choose_step(curves: Sequence[Sequence[tuple[int, float, float]]]) -> int:
    """The check step with the lowest held-out log-loss pooled over `curves` (rows (step, loss, ink)), each run's loss
    weighed by its held-out ink, so the pooled loss is that of every held px together.  Only steps every curve
    checked are candidates; ties go to the earlier step."""
    num: dict[int, float] = {}
    den: dict[int, float] = {}
    seen: dict[int, int] = {}
    for curve in curves:
        for step, loss, ink in curve:
            seen[step] = seen.get(step, 0) + 1
            if ink > 0:
                num[step] = num.get(step, 0.0) + loss * ink
                den[step] = den.get(step, 0.0) + ink
    candidates = [s for s in sorted(seen) if seen[s] == len(curves) and den.get(s, 0.0) > 0]
    if not candidates:
        raise ValueError("no check step with held-out ink common to every curve")
    return min(candidates, key=lambda s: num[s] / den[s])     # the first of equal losses: the earliest


def load_snapshot(path: Path) -> nn.Module:
    """A saved state dict as a net, in eval mode on cpu."""
    net = model.new_net()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.eval()


def _mean_chances(nets: Sequence[nn.Module], lum: np.ndarray, device: str, temperature: float,
                  each: Callable[[np.ndarray], None] | None = None) -> np.ndarray:
    """`model.prob(nets, lum, device, temperature)`, the same sums in the same order, but net by net: one (3, h, w)
    logits array alive at once.  `each` is given every net's own chances at `temperature` first."""
    total = None
    for net in nets:
        p = model.combine([model.logits(net, lum, device)], temperature)
        if each is not None:
            each(p)
        if total is None:
            total = p
        else:
            total += p
    total /= np.float32(len(nets))
    return total


def write_outputs(out: Path, sid: SheetId, p: np.ndarray, decision: decide.Decision, maps: int) -> None:
    """The page's four files (see the module): `p` its (3, h, w) chances, `decision` the cut made from them."""
    d = sheet_dir(out, sid)
    d.mkdir(parents=True, exist_ok=True)
    deleted = decision.deleted

    def replace(name: str, write) -> None:
        tmp = d / (name + ".tmp")
        write(tmp)
        os.replace(tmp, d / name)

    replace(OUTSIDE_PNG, lambda f: Image.fromarray(deleted).save(f, format="PNG"))
    q = np.round(np.clip(p[1] + p[2], 0.0, 1.0) * 255.0).astype(np.uint8)

    def save_q(f: Path) -> None:
        with open(f, "wb") as fh:
            np.savez_compressed(fh, q=q)
    replace(PROB_FILE, save_q)

    with Image.open(data.page_path(*sid, work=data.WORK)) as im:
        # 1-bit as 8-bit grey (paper's median may not be 0 or 1); a palette has no median of its indices
        a = np.array(im.convert({"1": "L", "P": "RGB"}.get(im.mode, im.mode)))
    if a.shape[:2] != deleted.shape:
        raise ValueError(f"{sid}: page is {a.shape[:2]}, the decision {deleted.shape}")
    kept = a[~deleted] if not deleted.all() else a.reshape(-1, *a.shape[2:])   # nothing kept: the whole page's
    a[deleted] = np.round(np.median(kept, axis=0)).astype(a.dtype)
    replace(ERASED_PNG, lambda f: Image.fromarray(a).save(f, format="PNG"))

    replace(OUTLINE_FILE, lambda f: f.write_text(json.dumps({"polygons": decide.outline(deleted), "maps": maps})))


def _interval(t: tuple[float, float, float]) -> dict:
    return {"mean": t[0], "lo": t[1], "hi": t[2]}


def _ratio_key(r: float) -> str:
    return f"{r:g}"


def _ink_share(where: np.ndarray, lum: np.ndarray) -> float:
    ink = 1.0 - lum.astype(np.float64)
    total = ink.sum()
    return float(ink[where].sum() / total) if total > 0 else float("nan")


def _baseline(held: Sequence[SheetId], pages: dict[SheetId, learn.Page], counts: dict[SheetId, int],
              new: dict[SheetId, dict]) -> dict:
    """The old two-pass model's held-out chances under `BASELINE`, for the held sheets it has: decided and scored as
    the new ones (no temperature), and the paired differences new - old of `COMPARED`."""
    old = {}
    for sid in held:
        f = sheet_dir(BASELINE, sid) / BASELINE_FILE
        if not f.exists():
            continue
        pg = pages[sid]
        with np.load(f) as z:
            inside = z["map"].astype(np.float32)
        if inside.shape != pg.lum.shape:
            log.warning("%s/%s: baseline is %s, the page %s; left out", *sid, inside.shape, pg.lum.shape)
            continue
        p = np.stack([inside, 1.0 - inside, np.zeros_like(inside)])
        s = scores.sheet_scores(p, decide.decide(p, pg.lum, counts[sid]).deleted, pg.cls, pg.lum)
        old[sid] = {k: s[k] for k in COMPARED}
    return {
        "dir": str(BASELINE),
        "sheets": [{"series": sid[0], "sheet": sid[1], **v} for sid, v in sorted(old.items())],
        "paired": {k: _interval(scores.paired_difference({sid: new[sid][k] for sid in old},
                                                         {sid: v[k] for sid, v in old.items()}, scores.BOOT_N))
                   for k in COMPARED} if old else {},
    }


def run_folds(labelled: Sequence[SheetId], recipe: learn.Recipe, device: str, out: Path = OUT,
              only: int | None = None) -> dict:
    """Score every labelled sheet held out (see the module); returns what it writes to `out/folds.json`.  `only` runs
    fold `only` with seed 0 alone, at the recipe's last check (no choice of step), and writes everything, its
    snapshots and summary included, under `out/fold-<only>/`.  The caller holds the GPU's turn."""
    if recipe.steps < recipe.check_every:
        raise ValueError(f"{recipe.steps} steps checked every {recipe.check_every}: no snapshot to choose")
    labelled = sorted(set(labelled))
    outer = folds.split(labelled, folds.OUTER_K)
    if only is None:
        fold_ids, seeds = list(range(len(outer))), SEEDS
    else:
        if not 0 <= only < len(outer):
            raise ValueError(f"fold {only} is not one of 0..{len(outer) - 1}")
        fold_ids, seeds, out = [only], SEEDS[:1], out / f"fold-{only}"
    (out / FOLDS_FILE).unlink(missing_ok=True)
    pages = labelled_pages(labelled)
    counts = data.map_counts(data.PAGE_KINDS)

    curves = []
    for k in fold_ids:
        fold = outer[k]
        for s in seeds:
            log.info("fold %d seed %d: train on %d sheets, hold out %s", k, s, len(fold.train),
                     ", ".join(f"{a}/{b}" for a, b in fold.held))
            _, info = learn.train([pages[sid] for sid in fold.train], dataclasses.replace(recipe, seed=s), device,
                                  held=[pages[sid] for sid in fold.held],
                                  snapshots=snapshot_path(out, k, s, 0).parent, label=f"fold {k} seed {s}")
            curves.append({"fold": k, "seed": s, **info})
    if only is None:
        step = choose_step([c["held_curve"] for c in curves])
    else:
        step = recipe.steps - recipe.steps % recipe.check_every
    log.info("step %d", step)

    fold_of = {sid: k for k in fold_ids for sid in outer[k].held}
    held = sorted(fold_of)

    def nets_of(sid: SheetId) -> list[nn.Module]:
        return [load_snapshot(snapshot_path(out, fold_of[sid], s, step)) for s in seeds]

    # pass 1: calibration px from each seed's logits; T fitted on every held sheet together
    rng = np.random.default_rng(0)
    samples = []
    for sid in held:
        pg = pages[sid]
        zs = [model.logits(net, pg.lum, device) for net in nets_of(sid)]   # sampled together: the same px
        samples.append(calibrate.sample(zs, pg.cls, pg.lum, rng, calibrate.N_CAL))
        del zs
    temperature = calibrate.fit_temperature(samples)
    t_lo, t_hi = calibrate.temperature_interval(samples, calibrate.CAL_BOOT)
    reliability = calibrate.reliability(samples, temperature)
    log.info("temperature %.3f (%.3f-%.3f)", temperature, t_lo, t_hi)

    # pass 2: the chances at T, the decision, the outputs and the scores
    sheets, new = [], {}
    for sid in held:
        pg = pages[sid]
        maps = counts[sid]
        own = []      # each seed's own decision at T, for the seed disagreement
        p = _mean_chances(nets_of(sid), pg.lum, device, temperature,
                          (lambda p_seed: own.append(decide.decide(p_seed, pg.lum, maps).deleted))
                          if len(seeds) > 1 else None)
        dec = decide.decide(p, pg.lum, maps)
        write_outputs(out, sid, p, dec, maps)
        sc = scores.sheet_scores(p, dec.deleted, pg.cls, pg.lum)
        ratios = {}
        for r in RATIOS_SHOWN:
            d = decide.decide(p, pg.lum, maps, decide.cut_level(r, decide.COST_MAP))
            ratios[_ratio_key(r)] = scores.sheet_scores(p, d.deleted, pg.cls, pg.lum, r, decide.COST_MAP)
        disagree = _ink_share(own[0] != own[1], pg.lum) if len(own) > 1 else None
        new[sid] = sc
        sheets.append({"series": sid[0], "sheet": sid[1], "fold": fold_of[sid], "maps": maps, "scores": sc,
                       "ratios": ratios, "seed_disagreement": disagree, "filled_ink": dec.filled_ink,
                       "dropped_ink": dec.dropped_ink, "removed_ink": dec.removed_ink})
        log.info("%s/%s: frame kept %.1f, map cut %.1f, cost %.1f, log-loss %.4f", *sid, sc["frame_kept"],
                 sc["map_cut"], sc["cost"], sc["log_loss"])

    pooled = {k: _interval(scores.bootstrap_mean([s["scores"][k] for s in sheets], scores.BOOT_N))
              for k in sheets[0]["scores"]}
    ratios = {}
    for r in RATIOS_SHOWN:
        key = _ratio_key(r)
        ratios[key] = {"cost_frame": r, "cost_map": decide.COST_MAP, "level": decide.cut_level(r, decide.COST_MAP),
                       "pooled": {k: _interval(scores.bootstrap_mean([s["ratios"][key][k] for s in sheets],
                                                                     scores.BOOT_N))
                                  for k in sheets[0]["ratios"][key]}}
    disagreement = (_interval(scores.bootstrap_mean([s["seed_disagreement"] for s in sheets], scores.BOOT_N))
                    if len(seeds) > 1 else None)
    result = {
        "recipe": dataclasses.asdict(recipe),
        "seeds": list(seeds),
        "step": step,
        "curves": curves,
        "temperature": temperature,
        "temperature_interval": [t_lo, t_hi],
        "reliability": reliability,
        "sheets": sheets,
        "pooled": pooled,
        "ratios": ratios,
        "seed_disagreement": disagreement,
        "baseline": _baseline(held, pages, counts, new),
    }
    _write_json(out / FOLDS_FILE, result)
    return result


def train_final(labelled: Sequence[SheetId], recipe: learn.Recipe, device: str, out: Path, step: int,
                temperature: float) -> None:
    """`SEEDS` trained on every labelled sheet for `step` steps (the held-out run's choice), saved with `temperature`
    (its fit) to `out/model/`; `model.json` is deleted first and written last, so a run that stops part way leaves no
    model to load.  The caller holds the GPU's turn."""
    labelled = sorted(set(labelled))
    pages = labelled_pages(labelled)
    d = out / MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").unlink(missing_ok=True)
    trained = dataclasses.replace(recipe, steps=step)
    for s in SEEDS:
        log.info("final seed %d: train on all %d sheets for %d steps", s, len(pages), step)
        net, _ = learn.train(list(pages.values()), dataclasses.replace(trained, seed=s), device,
                             label=f"final seed {s}")
        tmp = d / f"seed-{s}.pt.tmp"
        torch.save(net.state_dict(), tmp)
        os.replace(tmp, d / f"seed-{s}.pt")
    _write_json(d / "model.json", {"step": step, "temperature": temperature, "seeds": list(SEEDS),
                                   "sheets": [list(s) for s in labelled], "level": decide.cut_level(),
                                   "recipe": dataclasses.asdict(trained)})


def load_model(out: Path) -> tuple[list[nn.Module], float]:
    """The final nets (eval, cpu) and the temperature, from `out/model/`."""
    d = out / MODEL_DIR
    if not (d / "model.json").exists():
        raise FileNotFoundError(f"{d / 'model.json'} is missing: train the final model first (`sfzmaps frame`)")
    meta = json.loads((d / "model.json").read_text())
    return [load_snapshot(d / f"seed-{s}.pt") for s in meta["seeds"]], float(meta["temperature"])


def predict_pages(nets: Sequence[nn.Module], temperature: float, sids: Sequence[SheetId], device: str,
                  out: Path) -> dict[SheetId, dict]:
    """Each page of `sids` predicted with `nets` averaged at `temperature`, decided and written (`write_outputs`);
    returns each page's `filled_ink`, `dropped_ink` and `removed_ink`.  Each page is read once and let go."""
    counts = data.map_counts(data.PAGE_KINDS)
    stats = {}
    # Compressing a page's files takes longer than predicting it, and zlib releases the GIL, so the saves run on a
    # few threads while the device predicts the next page.  At most SAVE_THREADS pages wait in memory.
    with concurrent.futures.ThreadPoolExecutor(SAVE_THREADS) as pool:
        pending: list[concurrent.futures.Future] = []
        for i, sid in enumerate(sids, 1):
            lum = _lum(sid)
            p = _mean_chances(nets, lum, device, temperature)
            dec = decide.decide(p, lum, counts[sid])
            pending.append(pool.submit(write_outputs, out, sid, p, dec, counts[sid]))
            stats[sid] = {"filled_ink": dec.filled_ink, "dropped_ink": dec.dropped_ink,
                          "removed_ink": dec.removed_ink}
            log.info("%d/%d %s/%s: filled %.1f, dropped %.1f", i, len(sids), *sid, dec.filled_ink, dec.dropped_ink)
            while len(pending) >= SAVE_THREADS:
                pending.pop(0).result()
        for f in pending:
            f.result()
    return stats


def predict_all(device: str, out: Path = OUT) -> None:
    """Every map page that is not labelled, predicted with the final model; the labelled sheets keep their held-out
    outputs.  Then `PAGES_FILE`: every map page's filled and dropped ink (the labelled sheets' from `folds.json`,
    marked `held_out`), most dropped first.  The caller holds the GPU's turn."""
    torch.set_num_threads(learn.THREADS)
    torch.use_deterministic_algorithms(True)
    nets, temperature = load_model(out)
    labelled = set(data.labelled_sheets(data.LABELS))
    pages = data.map_pages(data.PAGE_KINDS, data.WORK)
    for sid in pages:
        if sid in labelled and not (sheet_dir(out, sid) / OUTSIDE_PNG).exists():
            log.warning("%s/%s is labelled but has no held-out outputs: run the folds", *sid)
    stats = predict_pages(nets, temperature, [sid for sid in pages if sid not in labelled], device, out)
    rows = [{"series": s, "sheet": h, "filled_ink": v["filled_ink"], "dropped_ink": v["dropped_ink"],
             "held_out": False} for (s, h), v in stats.items()]
    summary = out / FOLDS_FILE
    if summary.exists():
        on_page = set(pages)
        rows += [{"series": r["series"], "sheet": r["sheet"], "filled_ink": r["filled_ink"],
                  "dropped_ink": r["dropped_ink"], "held_out": True}
                 for r in json.loads(summary.read_text())["sheets"] if (r["series"], r["sheet"]) in on_page]
    else:
        log.warning("%s is missing: the labelled sheets are left out of %s", summary, PAGES_FILE)
    rows.sort(key=lambda r: -r["dropped_ink"])      # stable: equal drops keep page order
    _write_json(out / PAGES_FILE, rows)


def _fmt(iv: dict, digits: int = 1) -> str:
    return f"{iv['mean']:.{digits}f} [{iv['lo']:.{digits}f}, {iv['hi']:.{digits}f}]"


def format_scores(result: dict) -> str:
    """The held-out run for the owner: the step, T, the pooled scores at the cut with 90% sheet-bootstrap intervals,
    the trade-off at each shown ratio, the seed disagreement, the baseline and each sheet."""
    lo, hi = result["temperature_interval"]
    pooled = result["pooled"]
    lines = [
        f"held-out: {len(result['sheets'])} sheets, step {result['step']}, "
        f"temperature {result['temperature']:.3f} ({lo:.3f}-{hi:.3f})",
        f"cut at frame:map {decide.COST_FRAME:g}:{decide.COST_MAP:g} (delete where q > {decide.cut_level():.3f}); "
        "means over sheets [90% sheet-bootstrap interval]:",
        f"  frame ink kept  {_fmt(pooled['frame_kept'])}  share {_fmt(pooled['frame_kept_share'], 4)}",
        f"  map ink cut     {_fmt(pooled['map_cut'])}  share {_fmt(pooled['map_cut_share'], 4)}",
        f"  cost            {_fmt(pooled['cost'])}",
        f"  log-loss        {_fmt(pooled['log_loss'], 4)}",
        f"  HD95 px         {_fmt(pooled['hd95_px'])}",
        "other ratios (frame:map):",
    ]
    for key, r in result["ratios"].items():
        p = r["pooled"]
        lines.append(f"  {key}:1 (q > {r['level']:.3f})  frame kept {_fmt(p['frame_kept'])}, "
                     f"map cut {_fmt(p['map_cut'])}, cost {_fmt(p['cost'])}")
    sd = result["seed_disagreement"]
    lines.append("seed disagreement (ink share): " + (_fmt(sd, 4) if sd else "one seed"))
    base = result["baseline"]
    if base["sheets"]:
        lines.append(f"baseline (two-pass, {len(base['sheets'])} sheets), new - old: "
                     + ", ".join(f"{k} {_fmt(v)}" for k, v in base["paired"].items()))
    else:
        lines.append("baseline: no shared sheets")
    lines.append("per sheet:")
    for s in result["sheets"]:
        sc = s["scores"]
        lines.append(f"  {s['series']}/{s['sheet']} (fold {s['fold']}): frame kept {sc['frame_kept']:.1f}, "
                     f"map cut {sc['map_cut']:.1f}, cost {sc['cost']:.1f}, log-loss {sc['log_loss']:.4f}")
    return "\n".join(lines)
