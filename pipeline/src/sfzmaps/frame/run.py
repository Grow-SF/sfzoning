"""The learned frame finder's run: held-out folds, the final model, every map page.

`run_folds` scores every labelled sheet with networks that never saw it.  Each outer fold (`folds.split(labelled,
OUTER_K)`) holds out one sheet per series and works on the other 9 alone:

1. inner folds (`folds.split(train, INNER_K)`): the page pass trains on 6 and validates on 3; its prediction of those
   3 is their out-of-fold prior, so every one of the 9 gets a prior from a page pass that did not see it; the page
   pass's step count is chosen (`train.choose_steps`) over the 3 inner curves;
2. the page pass trains on all 9 for that step count;
3. the inner folds again: the pixel pass trains on 6 and validates on 3, each with its out-of-fold prior; the pixel
   pass's step count is chosen the same way;
4. the pixel pass trains on all 9, each with its out-of-fold prior, for that step count;
5. the fold's 3 held-out sheets are predicted with the fold's two networks and scored (`scores.sheet_scores`).

`train_final` trains on all 12 for the median of the folds' chosen step counts; the pixel pass's priors are each
sheet's held-out page-pass prediction, read back from its `map_prob.npz`.  `predict_all` writes every other map page's
prediction with the final model; the labelled sheets keep their held-out ones.

The page pass's prior is kept as float16, as it is saved: the pixel pass trains on, and predicts from, exactly the
`coarse` array in `map_prob.npz` (so `map` can be remade from the file).  Scores are of the float32 `map`, before it
is rounded to float16 for saving (float16 rounds a confident 0.99999 to 1.0, which would change the log-loss).

Outputs, all under `out` (default `data/frame`):
- `<series>/<sheet>/map_prob.npz`: `coarse` (page pass) and `map` (pixel pass), float16 map probabilities;
- `<series>/<sheet>/scores.json`: a labelled sheet's held-out `sheet_scores`;
- `folds.json`: per sheet its scores, fold and ink; per series and overall the ink-weighted means of each score (NaN
  scores skipped, with the count of sheets that contributed); each fold's curves and chosen step counts.  NaN and
  Infinity are written as Python's json writes them;
- `model/coarse.pt`, `model/fine.pt` (state dicts) and `model/recipe.json`.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch

from sfzmaps.frame import data, folds, predict, scores, train
from sfzmaps.frame.data import OUT, SheetId
from sfzmaps.frame.train import Page, Recipe

log = logging.getLogger(__name__)

PROB_FILE = "map_prob.npz"
SCORES_FILE = "scores.json"
FOLDS_FILE = "folds.json"
MODEL_DIR = "model"

# Each score's ink: the region whose ink it is a mean over, which weighs it in the means across sheets (so a mean is
# the pooled score over every sheet's ink).  The edge distances are not ink means; they are weighed by all ink.
REGION_OF = {
    "log_loss_frame_band": "frame_band",
    "log_loss_map_band": "map_band",
    "log_loss_all_ink": "all",
    "frame_called_map": "frame_band",
    "map_called_frame": "map_band",
    "edge_median_px": "all",
    "edge_hd95_px": "all",
}


def sheet_dir(out: Path, sid: SheetId) -> Path:
    return out / sid[0] / sid[1]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def save_prob(out: Path, sid: SheetId, coarse: np.ndarray, prob: np.ndarray) -> Path:
    d = sheet_dir(out, sid)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "map_prob.tmp.npz"
    np.savez_compressed(tmp, coarse=coarse.astype(np.float16), map=prob.astype(np.float16))
    os.replace(tmp, d / PROB_FILE)
    return d / PROB_FILE


def load_prob(out: Path, sid: SheetId) -> dict[str, np.ndarray]:
    path = sheet_dir(out, sid) / PROB_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the folds first (`sfzmaps frame --only-folds`)")
    with np.load(path) as z:
        return {"coarse": z["coarse"], "map": z["map"]}


def _lum(sid: SheetId) -> np.ndarray:
    return data.load_luminance(data.page_path(*sid, work=data.WORK))


def _labelled_page(sid: SheetId) -> Page:
    lum = _lum(sid)
    mask = data.map_mask(data.load_label(*sid, root=data.LABELS), lum.shape)
    return Page(id=sid, lum=lum, mask=mask, prior=None)


def _prior(coarse_net, lum: np.ndarray, device: str) -> np.ndarray:
    """The page pass's map probability as saved and as the pixel pass reads it: float16."""
    return predict.predict_coarse(coarse_net, lum, device).astype(np.float16)


def _predict(coarse_net, fine_net, lum: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """(coarse float16, map float32): the pixel pass reads the float16 prior that is saved."""
    coarse = _prior(coarse_net, lum, device)
    return coarse, predict.predict_tiles(fine_net, lum, coarse, device)


def _ink(mask: np.ndarray, lum: np.ndarray) -> dict[str, float]:
    """The ink each score is a mean over (`REGION_OF`)."""
    w = data.ink_weight(lum).astype(np.float64)
    frame_band, map_band = scores.bands(mask)
    return {"frame_band": float(w[frame_band].sum()), "map_band": float(w[map_band].sum()), "all": float(w.sum())}


def means(sheets: list[dict]) -> dict[str, dict]:
    """Per score, the mean over `sheets` weighed by each sheet's ink for that score ({"mean", "sheets"}: the count of
    sheets that contributed).  A NaN score (a band with no ink) is skipped; an infinite one (no 1/2 contour) is
    kept, so the mean says so."""
    out = {}
    for key, region in REGION_OF.items():
        vals = [(s["scores"][key], s["ink"][region]) for s in sheets if not math.isnan(s["scores"][key])]
        total = sum(w for _, w in vals)
        mean = sum(v * w for v, w in vals) / total if total > 0 else float("nan")
        out[key] = {"mean": mean, "sheets": len(vals)}
    return out


def _held_once(inner: list[folds.Fold], train_ids: tuple[SheetId, ...]) -> None:
    held = [s for f in inner for s in f.held]
    if sorted(held) != sorted(train_ids):
        raise ValueError("the inner folds must hold out each training sheet exactly once, so each has one "
                         "out-of-fold prior")


def _fold(fold: folds.Fold, pages: dict[SheetId, Page], recipe: Recipe, device: str):
    """Steps 1-4 of the module for one outer fold: ({curves and chosen steps}, (page-pass net, pixel-pass net)).
    Nothing here sees `fold.held`."""
    inner = folds.split(list(fold.train), folds.INNER_K)
    _held_once(inner, fold.train)
    priors, coarse_curves = {}, []
    for i, f in enumerate(inner):
        log.info("  page pass, inner fold %d: train %d, validate %d", i, len(f.train), len(f.held))
        net, rec = train.train_coarse([pages[s] for s in f.train], recipe, device, val=[pages[s] for s in f.held])
        coarse_curves.append(rec["val_curve"])
        for s in f.held:
            priors[s] = _prior(net, pages[s].lum, device)
        del net
    coarse_steps = train.choose_steps(coarse_curves)
    log.info("  page pass on all %d for %d steps", len(fold.train), coarse_steps)
    coarse_net, _ = train.train_coarse([pages[s] for s in fold.train],
                                       dataclasses.replace(recipe, steps=coarse_steps), device)

    def with_prior(sids):
        return [dataclasses.replace(pages[s], prior=priors[s]) for s in sids]

    fine_curves = []
    for i, f in enumerate(inner):
        log.info("  pixel pass, inner fold %d: train %d, validate %d", i, len(f.train), len(f.held))
        _, rec = train.train_fine(with_prior(f.train), recipe, device, val=with_prior(f.held))
        fine_curves.append(rec["val_curve"])
    fine_steps = train.choose_steps(fine_curves)
    log.info("  pixel pass on all %d for %d steps", len(fold.train), fine_steps)
    fine_net, _ = train.train_fine(with_prior(fold.train), dataclasses.replace(recipe, steps=fine_steps), device)
    info = {"coarse_steps": coarse_steps, "fine_steps": fine_steps,
            "coarse_curves": coarse_curves, "fine_curves": fine_curves}
    return info, (coarse_net, fine_net)


def run_folds(labelled: list[SheetId], recipe: Recipe, device: str, out: Path = OUT,
              only: int | None = None) -> dict:
    """Score every labelled sheet held out (see the module); returns what it writes to `out/folds.json`.  `only` runs
    outer fold `only` alone (the determinism check), writing `out/fold-<only>.json` instead."""
    labelled = sorted(set(labelled))
    outer = folds.split(labelled, folds.OUTER_K)
    if only is not None and not 0 <= only < len(outer):
        raise ValueError(f"fold {only} is not one of 0..{len(outer) - 1}")
    _held_once(outer, tuple(labelled))
    pages = {sid: _labelled_page(sid) for sid in labelled}
    sheets, fold_info = [], []
    for k, fold in enumerate(outer):
        if only is not None and k != only:
            continue
        log.info("outer fold %d: hold out %s", k, ", ".join(f"{s}/{h}" for s, h in fold.held))
        info, (coarse_net, fine_net) = _fold(fold, pages, recipe, device)
        for sid in fold.held:
            pg = pages[sid]
            coarse, prob = _predict(coarse_net, fine_net, pg.lum, device)
            save_prob(out, sid, coarse, prob)
            s = scores.sheet_scores(prob, pg.mask, pg.lum)
            _write_json(sheet_dir(out, sid) / SCORES_FILE, s)
            sheets.append({"series": sid[0], "sheet": sid[1], "fold": k, "scores": s, "ink": _ink(pg.mask, pg.lum)})
            log.info("  %s/%s: held-out log-loss %.4f over all ink", sid[0], sid[1], s["log_loss_all_ink"])
        fold_info.append({"fold": k, "held": [list(s) for s in fold.held], **info})
    result = {
        "recipe": dataclasses.asdict(recipe),
        "sheets": sheets,
        "series": {name: means([s for s in sheets if s["series"] == name])
                   for name in sorted({s["series"] for s in sheets})},
        "overall": means(sheets),
        "folds": fold_info,
        "steps": {"coarse": [f["coarse_steps"] for f in fold_info], "fine": [f["fine_steps"] for f in fold_info]},
    }
    _write_json(out / (FOLDS_FILE if only is None else f"fold-{only}.json"), result)
    return result


def _median_steps(steps: list[int]) -> int:
    if not steps:
        raise ValueError("no fold's chosen step count to take the median of")
    return int(round(float(np.median(steps))))


def train_final(labelled: list[SheetId], recipe: Recipe, device: str, steps: dict, out: Path = OUT) -> None:
    """The final model: both passes on every labelled sheet, for the median of the folds' chosen step counts
    (`steps` = {"coarse": [...], "fine": [...]}, as `run_folds` returns them); the pixel pass's priors are each
    sheet's held-out page-pass prediction from `run_folds`.  Saves `out/model/`."""
    labelled = sorted(set(labelled))
    coarse_steps, fine_steps = _median_steps(steps["coarse"]), _median_steps(steps["fine"])
    pages = [_labelled_page(sid) for sid in labelled]
    priors = {}
    for pg in pages:
        prior = load_prob(out, pg.id)["coarse"]
        if prior.shape != pg.lum.shape:
            raise ValueError(f"{pg.id}: held-out prior {prior.shape} is not the page's shape {pg.lum.shape}")
        priors[pg.id] = prior
    log.info("final page pass on all %d for %d steps", len(pages), coarse_steps)
    coarse_net, _ = train.train_coarse(pages, dataclasses.replace(recipe, steps=coarse_steps), device)
    log.info("final pixel pass on all %d for %d steps", len(pages), fine_steps)
    fine_net, _ = train.train_fine([dataclasses.replace(pg, prior=priors[pg.id]) for pg in pages],
                                   dataclasses.replace(recipe, steps=fine_steps), device)
    model = out / MODEL_DIR
    model.mkdir(parents=True, exist_ok=True)
    for name, net in (("coarse.pt", coarse_net), ("fine.pt", fine_net)):
        tmp = model / (name + ".tmp")
        torch.save(net.state_dict(), tmp)
        os.replace(tmp, model / name)
    _write_json(model / "recipe.json", {"recipe": dataclasses.asdict(recipe), "coarse_steps": coarse_steps,
                                        "fine_steps": fine_steps, "fold_steps": steps,
                                        "sheets": [list(s) for s in labelled]})


def load_model(out: Path = OUT):
    """(page-pass net, pixel-pass net) from `out/model/`, in eval mode on cpu."""
    nets = []
    for name, make in (("coarse.pt", predict.new_coarse_net), ("fine.pt", predict.new_fine_net)):
        net = make()
        net.load_state_dict(torch.load(out / MODEL_DIR / name, map_location="cpu", weights_only=True))
        nets.append(net.eval())
    return tuple(nets)


def predict_all(device: str, out: Path = OUT) -> None:
    """Every map page that is not labelled, predicted with the final model into its `map_prob.npz`; each page read
    once and let go.  The labelled sheets keep their held-out predictions."""
    torch.set_num_threads(train.THREADS)
    torch.use_deterministic_algorithms(True)
    coarse_net, fine_net = load_model(out)
    labelled = set(data.labelled_sheets(data.LABELS))
    pages = data.map_pages(data.PAGE_KINDS, data.WORK)
    for i, sid in enumerate(pages, 1):
        if sid in labelled:
            if not (sheet_dir(out, sid) / PROB_FILE).exists():
                log.warning("%s/%s is labelled but has no held-out prediction: run the folds", *sid)
            continue
        coarse, prob = _predict(coarse_net, fine_net, _lum(sid), device)
        save_prob(out, sid, coarse, prob)
        log.info("%d/%d %s/%s", i, len(pages), *sid)


def format_scores(result: dict) -> str:
    """The overall and per-series means, one line each."""
    keys = list(REGION_OF)
    rows = [("overall", result["overall"])] + sorted(result["series"].items())
    lines = ["held-out scores (ink-weighted means; n = sheets with a score)"]
    for name, m in rows:
        lines.append(f"{name:>22}  " + "  ".join(f"{k} {m[k]['mean']:.4f} (n={m[k]['sheets']})" for k in keys))
    return "\n".join(lines)
