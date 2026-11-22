"""Each lot's district from the learned reader (ruling R8): the model's own pooled lot probability, the objective it was
trained on (`train.lot_logprob`, before the read-noise matrix):

    p(j) = sum_px softmax_lot(weight)_px softmax(districts_px)(j)

over ALL the lot's px on every sheet it lies on, pooled once.

  - every sheet is predicted whole first (`sheet_outputs`, one pass over the page as `train._predict` gives it: no tile
    edge splits a lot), then pooled: per sheet and lot, log sum_px exp(weight) and log sum_px exp(weight) p_px(j)
    (`sheet_sums`), added over the sheets in log space (`pool`).  That is exactly the one pooling over the lot's px
    wherever they lie: a lot on two sheets is never the mean of two sheet pools;
  - each quarter's lots are predicted by the fold that withheld that quarter (`predict_folds`, `by_fold`): answers_<q>
    for with_answers.parquet, reads_<q> for reads_only.parquet.  A fold predicts only the sheets that hold px of its
    quarter's lots and pools only those lots.  all.parquet: the all-quarters model (answers_all) on every lot, for
    pictures only; it is never scored (it trained on every quarter);
  - a lot with no px on any sheet (outside every sheet's area) has no row.

A row (`lot_districts`): lot, blklot, quarter, code (the most probable district), p (its probability), p_<district>
(the six, model.NAMES order), px (the lot's px over every sheet), n_sheets (how many sheets they lie on); the fold
outputs add model (the run that predicted the lot).

Writes under data/solver/experiments/hatch-cnn/predict/ (`run`): with_answers.parquet, reads_only.parquet, all.parquet
and summary.json; then the pictures into look/ (`look_sheets`: each sheet's lots in their predicted district, opacity
its probability; `look_random`: the contact sheet of the owner's random lots).  No score against the owner's answers is
computed here: the owner looks first (Task 5 scores).

Deterministic: torch's deterministic algorithms (`train.settle_torch`); on the Mac's GPU the run holds its exclusive
turn (`sfzmaps.gpu`).  $0: no API calls.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sfzmaps import gpu
from sfzmaps.solver import paths
from sfzmaps.solver.hatchnet import data, model, train

log = logging.getLogger(__name__)

ROOT = data.EXPERIMENT / "predict"
OUTPUTS = ("with_answers", "reads_only", "all")
K = len(model.NAMES)
P_COLUMNS = [f"p_{c}" for c in model.NAMES]
# The sheets drawn (`look_sheets`) -> the zoom's top-left (y, x), upright scan px.
LOOK = {"20180102094149543": train.WINDOWS["20180102094149543"],      # dense
        "20180102100525362": train.WINDOWS["20180102100525362"],      # south
        "20180102094317284": (2290, 1750),     # Islais / Bayview: the diagonal checkerboard (U) blocks and the 1R
        "20180102094355582": (600, 950)}       # Lake Merced: the hatched row along the sheet's top and the curved lots
CONTACT_STRATUM = "random"           # the owner's random page's random lots (`look_random`)

assert model.WEIGHT == model.DISTRICTS.stop, "the weight logit follows the six district logits"


# --- pooling --------------------------------------------------------------------------------------------------------

def sheet_sums(lot, district_logits, weight_logits, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One sheet's part of every lot's pooling, float64: lz (n,) = log sum_px exp(weight), ls (n, 6) =
    log sum_px exp(weight) softmax(districts_px), and the px counted (n,), over the px with `lot` >= 0 (-inf, 0 for a
    lot with none here).  `lot` (any shape) each px's lot 0..n-1 or -1; `district_logits` (6, *lot.shape);
    `weight_logits` (lot.shape)."""
    lot = np.asarray(lot).reshape(-1)
    m = lot >= 0
    if m.any() and int(lot[m].max()) >= n:
        raise ValueError(f"lot {int(lot[m].max())} is not in the table of {n}")
    idx = torch.from_numpy(lot[m].astype(np.int64))
    a = torch.from_numpy(np.asarray(weight_logits).reshape(-1)[m].astype(np.float64))
    d = torch.from_numpy(np.ascontiguousarray(np.asarray(district_logits).reshape(K, -1)[:, m].T, np.float64))
    lz = train._segment_lse(a, idx, n)
    ls = train._segment_lse(a[:, None] + F.log_softmax(d, -1), idx, n)
    return lz.numpy(), ls.numpy(), np.bincount(lot[m], minlength=n)


def pool(parts: Iterable, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every lot's log district probabilities (n, 6) from the sheets' parts (`sheet_sums`), added in log space: the
    pooling over all the lot's px at once (NaN rows for a lot with no px); its px over every sheet; and how many sheets
    they lie on."""
    lz, ls = np.full(n, -np.inf), np.full((n, K), -np.inf)
    px, sheets = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for z, s, c in parts:
        lz, ls = np.logaddexp(lz, z), np.logaddexp(ls, s)
        px += c
        sheets += c > 0
    with np.errstate(invalid="ignore"):
        lp = ls - lz[:, None]
    lp[px == 0] = np.nan
    return lp, px, sheets


def lot_districts(sheets: Iterable, lots: pd.DataFrame) -> pd.DataFrame:
    """Each lot's pooled district (see the module) from `sheets`, an iterable of (lot, district_logits, weight_logits)
    per sheet (`sheet_sums`; consumed one at a time), and the lot table (row i is lot i; blklot, quarter).  A lot with
    no px on any sheet has no row."""
    n = len(lots)
    if not np.array_equal(lots["lot"].to_numpy(), np.arange(n)):
        raise ValueError("the lot table's rows must be lots 0..n-1 in order")
    lp, px, n_sheets = pool((sheet_sums(lot, d, w, n) for lot, d, w in sheets), n)
    keep = px > 0
    p = np.exp(lp[keep])
    out = pd.DataFrame({"lot": lots["lot"].to_numpy()[keep], "blklot": lots["blklot"].to_numpy()[keep],
                        "quarter": lots["quarter"].to_numpy()[keep],
                        "code": np.array(model.NAMES, object)[p.argmax(1)], "p": p.max(1)})
    for j, c in enumerate(P_COLUMNS):
        out[c] = p[:, j]
    out["px"], out["n_sheets"] = px[keep], n_sheets[keep]
    return out


# --- the models -----------------------------------------------------------------------------------------------------

def sheet_outputs(net: model.HatchNet, sheet: data.SheetRasters, tile: int = train.PRED_TILE) -> np.ndarray:
    """(7, h, w) float32: the six district logits and the weight logit at every px of the sheet's page, one pass over
    the whole page (`train._predict`)."""
    return train._predict(net, sheet, tile, lambda t: t[0, model.DISTRICTS.start:model.WEIGHT + 1], K + 1)


def predict_model(net: model.HatchNet, rasters: list[data.SheetRasters], lots: pd.DataFrame,
                  quarter: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """The lots of `quarter` (None: every lot) pooled from the network's outputs on every sheet that holds px of them
    (`lot_districts`); and those sheets, in order."""
    mine = np.ones(len(lots), bool) if quarter is None else (lots["quarter"].to_numpy() == quarter)
    used = []

    def sheets():
        for r in rasters:
            on = r.lot >= 0
            on[on] = mine[r.lot[on]]
            if not on.any():
                continue
            used.append(r.sheet)
            o = sheet_outputs(net, r)
            yield np.where(on, r.lot, -1), o[:K], o[K]

    return lot_districts(sheets(), lots), used


def by_fold(frames: dict[str, pd.DataFrame], with_answers: bool) -> pd.DataFrame:
    """Each quarter q's lots from the fold that withheld q (`train.run_name(q, with_answers)`) and from no other run,
    with the run's name (model), in lot order."""
    parts = []
    for q in paths.QUARTERS:
        name = train.run_name(q, with_answers)
        f = frames[name]
        parts.append(f[f["quarter"] == q].assign(model=name))
    return pd.concat(parts, ignore_index=True).sort_values("lot", kind="stable").reset_index(drop=True)


def predict_folds(load: Callable[[str], torch.nn.Module], rasters: list[data.SheetRasters], lots: pd.DataFrame,
                  device: str | None = None, runs=train.RUNS) -> tuple[dict[str, pd.DataFrame], dict]:
    """The three outputs ({with_answers, reads_only, all}: `by_fold` of the folds, and the all-quarters model on every
    lot) from the runs `runs` ((q, with_answers), `train.RUNS`), each network got by `load(run name)` one at a time; and
    per run, the sheets it predicted, its rows and its seconds."""
    device = device or train.default_device()
    frames, notes = {}, {}
    for q, with_answers in runs:
        name = train.run_name(q, with_answers)
        t0 = time.perf_counter()
        net = load(name).to(device).eval()
        frames[name], used = predict_model(net, rasters, lots, q)
        del net
        notes[name] = {"sheets": used, "rows": len(frames[name]), "seconds": round(time.perf_counter() - t0, 1)}
        log.info("%s: %d lots from %d sheets, %.0f s", name, len(frames[name]), len(used), notes[name]["seconds"])
    out = {"with_answers": by_fold(frames, True), "reads_only": by_fold(frames, False),
           "all": frames[train.DELIVERABLE].assign(model=train.DELIVERABLE)}
    return out, notes


def load_rasters(series: str) -> list[data.SheetRasters]:
    """The written rasters of `series` (`data.data_dir`), without the street labels."""
    files = sorted(data.data_dir(series).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no rasters under {data.data_dir(series)}: run `hatchnet data --series {series}`")
    return [dataclasses.replace(data.read_rasters(f), street=None, centre=None, street_w=None) for f in files]


def _counts(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def run(series: str, root: Path | None = None, models: Path | None = None, device: str | None = None,
        look: bool = True) -> dict:
    """Predict every lot with the nine runs (`predict_folds`, on the GPU's exclusive turn) from `models` (default
    hatch-cnn/districts/models), write the three outputs and summary.json under `root` (default ROOT), then the
    pictures (`look_sheets`, `look_random`)."""
    root = Path(root) if root is not None else ROOT
    models = Path(models) if models is not None else train.DISTRICTS_ROOT / "models"
    device = device or train.default_device()
    train.settle_torch()
    folder = data.data_dir(series)
    lots = pd.read_parquet(folder / "lots.parquet")
    rasters = load_rasters(series)
    runs = {train.run_name(q, w): (q, w) for q, w in train.RUNS}
    saved_notes = {}

    def load(name: str) -> torch.nn.Module:
        f = models / f"{name}.pt"
        net, saved = train.load_model(f, device)
        if (saved.get("fold"), saved.get("with_answers")) != runs[name]:
            raise ValueError(f"{f}: fold {saved.get('fold')}, with answers {saved.get('with_answers')}; "
                             f"{name} is {runs[name]}")
        saved_notes[name] = {"model": str(f), "step": saved["step"], "state_sha256": train.state_digest(
            saved["state_dict"]), "device_trained": saved.get("device")}
        return net

    t0 = time.perf_counter()
    with gpu.gpu_turn(True, device):
        out, notes = predict_folds(load, rasters, lots, device)
    root.mkdir(parents=True, exist_ok=True)
    for name, f in out.items():
        f.to_parquet(root / f"{name}.parquet", index=False)
    rowless = np.setdiff1d(lots["lot"].to_numpy(), out["all"]["lot"].to_numpy())
    summary = {
        "series": series, "device": device, "torch": torch.__version__, "names": model.NAMES,
        "seconds": round(time.perf_counter() - t0, 1),
        "runs": {n: {**saved_notes[n], **notes[n]} for n in notes},
        "outputs": {n: {"file": str(root / f"{n}.parquet"), "rows": len(f),
                        "rows_per_model": _counts(f["model"]), "lots_by_sheets": _counts(f["n_sheets"])}
                    for n, f in out.items()},
        "lots": {"total": len(lots), "with_a_row": int(len(lots) - len(rowless)),
                 "without_a_row": {"count": int(len(rowless)),
                                   "no_px_on_any_sheet": int((lots["px"].to_numpy()[rowless] == 0).sum()),
                                   "read": int((lots["read_code"].to_numpy()[rowless] != "").sum()),
                                   "answered": int((lots["answer"].to_numpy()[rowless] != "").sum()),
                                   "no_centre_sheet": int((lots["centre_sheet"].to_numpy()[rowless] == "").sum()),
                                   "blklots": lots["blklot"].to_numpy()[rowless].tolist()}},
        "note": "all.parquet is the all-quarters model's (trained on every quarter): for pictures only, never scored"}
    for n in OUTPUTS:
        if len(out[n]) != summary["lots"]["with_a_row"]:
            raise ValueError(f"{n}: {len(out[n])} rows, {summary['lots']['with_a_row']} lots have px")
    (root / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    if look:
        summary["pictures"] = [str(p) for p in look_sheets(series, root, out["with_answers"], rasters=rasters)]
        summary["pictures"] += [str(p) for p in look_random(series, root, out, lots, rasters)]
        (root / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def read_outputs(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = Path(root) if root is not None else ROOT
    return {n: pd.read_parquet(root / f"{n}.parquet") for n in OUTPUTS}


# --- the pictures ---------------------------------------------------------------------------------------------------

def _per_lot(frame: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(n,) each lot's predicted district index (-1 no row) and its probability."""
    code, prob = np.full(n, -1, np.int64), np.zeros(n, np.float32)
    lot = frame["lot"].to_numpy()
    code[lot] = [train.CODE[c] for c in frame["code"]]
    prob[lot] = frame["p"].to_numpy()
    return code, prob


def look_sheets(series: str, root: Path | None = None, frame: pd.DataFrame | None = None, sheets=LOOK,
                which: str = "with_answers", rasters: list[data.SheetRasters] | None = None) -> list[Path]:
    """Each sheet of `sheets` with every lot in its predicted district at opacity = its probability, from the fold
    predictions `frame` (default: root/<which>.parquet), whole and zoomed, with the seams between the lots' quarters
    (each side predicted by another fold's model), into root/look (`look.district_lots`)."""
    from sfzmaps.solver.hatchnet import look
    root = Path(root) if root is not None else ROOT
    frame = frame if frame is not None else pd.read_parquet(root / f"{which}.parquet")
    lots = pd.read_parquet(data.data_dir(series) / "lots.parquet")
    n = len(lots)
    code, prob = _per_lot(frame, n)
    quarter = np.array([paths.QUARTERS.index(q) if q in paths.QUARTERS else -1 for q in lots["quarter"]], np.int64)
    have = {r.sheet: r for r in rasters or []}
    kind = "with answers (answers_<q>)" if which == "with_answers" else "reads-only (reads_<q>)"
    legend = [f"each lot: its predicted district, by the fold that withheld its quarter, {kind}",
              "pooled over all its px on every sheet with the learned weights; opacity = its probability",
              "cyan: the seam between the lots' quarters (each side another fold's model)",
              "px in no lot (streets; areas another sheet labels): as scanned; grey: off the frame cut"]
    out = []
    for sheet, at in sheets.items():
        r = have.get(sheet) or data.read_rasters(data.data_dir(series) / f"{sheet}.npz")
        on = r.lot >= 0
        lot = np.maximum(r.lot, 0)
        here = np.unique(r.lot[on])
        here = here[code[here] >= 0]
        shares = np.bincount(code[here], minlength=K) / max(len(here), 1)
        seams = look.quarter_seams(r, np.where(on, quarter[lot], -1))
        note = f"fold predictions, {kind}"
        out += look.district_lots(r, np.where(on, code[lot], -1), np.where(on, prob[lot], 0.0), root / "look", at,
                                  note, shares, len(here), legend=legend, name=f"lots_{which}", seams=seams)
    return out


def contact_order(t: pd.DataFrame) -> pd.Index:
    """The order of the contact sheet's tiles (`t`'s index): the lots whose predicted code differs from the owner's
    answer first, each group surest first (p), ties by blklot."""
    key = pd.DataFrame({"same": t["code"].to_numpy() == t["answer"].to_numpy(), "p": -t["p"].to_numpy(),
                        "blklot": t["blklot"].to_numpy()}, index=t.index)
    return key.sort_values(["same", "p", "blklot"], kind="stable").index


def random_lots(lots: pd.DataFrame, out: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The owner's random lots (stratum CONTACT_STRATUM) with the reader's read, the owner's answer, the prediction
    with answers (code, p, model, n_sheets) and the reads-only one (code_reads, p_reads), in `contact_order`."""
    t = lots[lots["stratum"] == CONTACT_STRATUM][["lot", "blklot", "quarter", "read_code", "read_conf", "answer"]]
    w = out["with_answers"][["lot", "code", "p", "model", "n_sheets"]]
    ro = out["reads_only"][["lot", "code", "p"]].rename(columns={"code": "code_reads", "p": "p_reads"})
    t = t.merge(w, on="lot", how="left").merge(ro, on="lot", how="left")
    if t["code"].isna().any():
        raise ValueError(f"random lots with no prediction (no px): {t.loc[t['code'].isna(), 'blklot'].tolist()}")
    return t.loc[contact_order(t)].reset_index(drop=True)


def lot_boxes(rasters: list[data.SheetRasters], ids: np.ndarray, n: int) -> dict[int, tuple]:
    """For each lot of `ids`: (sheet index, its bounding box there (slice, slice), its px on each sheet) on the sheet
    that holds most of its px (ties: the first sheet in name order)."""
    from scipy import ndimage
    where = np.zeros(n, np.int64)
    where[ids] = np.arange(1, len(ids) + 1)
    counts = np.zeros((len(rasters), len(ids) + 1), np.int64)
    boxes = []
    for i, r in enumerate(rasters):
        lab = np.where(r.lot >= 0, where[np.maximum(r.lot, 0)], 0)
        counts[i] = np.bincount(lab.ravel(), minlength=len(ids) + 1)
        boxes.append(ndimage.find_objects(lab, max_label=len(ids)))
    got = {}
    for k, lot in enumerate(ids):
        c = counts[:, k + 1]
        if not c.any():
            continue
        i = int(np.argmax(c))
        got[int(lot)] = (i, boxes[i][k], c)
    return got


def look_random(series: str, root: Path | None = None, out: dict[str, pd.DataFrame] | None = None,
                lots: pd.DataFrame | None = None, rasters: list[data.SheetRasters] | None = None) -> list[Path]:
    """The contact sheet of the owner's random lots (`random_lots`), into root/look/contact_random_<k>.png
    (`look.contact_pages`): per lot a scan crop with the lot outlined, the prediction with answers (code, p, its
    model), the reads-only one, the reader's read and the owner's answer; disagreements with the answer first."""
    from sfzmaps.solver.hatchnet import look
    root = Path(root) if root is not None else ROOT
    out = out if out is not None else read_outputs(root)
    lots = lots if lots is not None else pd.read_parquet(data.data_dir(series) / "lots.parquet")
    rasters = rasters if rasters is not None else load_rasters(series)
    t = random_lots(lots, out)
    boxes = lot_boxes(rasters, t["lot"].to_numpy(), len(lots))
    tiles = []
    for _, row in t.iterrows():
        i, box, c = boxes[int(row["lot"])]
        r = rasters[i]
        crop, scale = look.lot_crop(r, int(row["lot"]), box)
        more = f" +{int((c > 0).sum()) - 1}" if (c > 0).sum() > 1 else ""
        lines = [f"{row['blklot']}  {row['quarter']}  {r.sheet[8:]}{more}  {scale}",
                 f"pred {row['code']} {row['p']:.2f}  ({row['model']})",
                 f"reads-only {row['code_reads']} {row['p_reads']:.2f}",
                 (f"read {row['read_code']} {row['read_conf']:.2f}" if row["read_code"] else "read -")
                 + f"   answer {row['answer']}"]
        tiles.append(look.contact_tile(crop, lines, row["code"] != row["answer"]))
    head = [f"{series}: the owner's random lots ({len(t)}), each predicted by the fold that withheld its quarter",
            "red frame: the prediction (with answers) differs from the owner's answer; these first, then the rest; "
            "each group surest first",
            "pred: with answers (answers_<q>)  reads-only: reads_<q>  read: the pattern reader (its confidence)  "
            "answer: the owner's",
            "magenta: the lot's outline on the sheet shown (its px there; +n: px on n more sheets)  x2 / x1: scan px "
            "zoom, 1/k: shrunk to fit"]
    return look.contact_pages(tiles, root / "look", f"contact_{CONTACT_STRATUM}", head)
