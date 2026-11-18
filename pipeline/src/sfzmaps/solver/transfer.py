"""The 2000 map's fitted models, applied to another series: one model learned from the April 2000 table, used on
every year (the owner's design).  Another series has no table to fit on (the 1996-2000 scan is scored on the 2000
table, so fitting on it would read the answers), so each of its fits takes the 2000 map's instead:

  walls      the ridge logistic weights of each held-out quarter, fitted on the 2000 page with TRANSFER_WALLS: the
             detector's chance of text in place of Terra's letter share (`wallevidence.TEXT_WEIGHTED`), so a series
             no Terra read gets the same features; applied to the series' own border features
  trust      the trust of each reading, fitted on the 2000 prints with only the reads the series has (READ_SETS:
             Sol's alone, or Sol's and Terra's; the third read, its re-check and the turned pass were asked of the
             2000 sheets only), applied to the series' readings
  streetside the four rates of each held-out quarter, from the same masked 2000 readings
  lotsize    the 2000 table's lot-size evidence (a code of another era carries none)
  placement  the four held-out conditional logits, fitted on the 2000 labels (`placement.fit_model`)

`build_models` (run on the 2000 map: `sfzmaps solver transfer-models`) writes them into paths.TRANSFER_DIR; the
series' stages read them (`sfzmaps solver --series S labels`, `walls`, `solve`)."""
import json
import os
import logging
import shutil
import time

import numpy as np

from sfzmaps.solver import paths

log = logging.getLogger(__name__)

TRANSFER_WALLS = ("text_weighted", "street_crossing", "context")
READ_SETS = {"sol": ("sol_code",), "sol+terra": ("sol_code", "terra_code")}


def mask_reads(prints, reads):
    """The prints as a series with only the reads `reads` (columns of trust.READS) would have them: every other read
    blank (with its confidence, found and leader flags), Terra's box gone without Terra, and the prints left with no
    read dropped.  Returns the masked prints and the kept rows' positions in `prints`."""
    from sfzmaps.solver import trust
    p = prints.copy()
    for col in trust.READS:
        if col not in reads:
            p[col] = ""
    if "terra_code" not in reads:
        for c in ("box_x0", "box_y0", "box_x1", "box_y1"):
            p[c] = np.nan
    p["third_confidence"] = np.nan
    p["third_found"] = False
    p["third_leader"] = False
    p["recheck_asked"] = False
    p["recheck_found"] = False
    p["recheck_confidence"] = np.nan
    p["turned_edge"] = np.nan
    has = np.zeros(len(p), bool)
    for col in reads:
        has |= (p[col].astype(str) != "").to_numpy()
    kept = np.flatnonzero(has)
    p = p.iloc[kept].reset_index(drop=True)
    p["kind"] = np.where(p["terra_code"].astype(str) != "", np.where(p["sol_code"].astype(str) != "", 0, 1),
                         2).astype("int8")
    return p, kept


def read_set(prints) -> str:
    """The READ_SETS name of a series' prints: Sol's and Terra's when Terra read any of them, else Sol's alone."""
    return "sol+terra" if (prints["terra_code"].astype(str) != "").any() else "sol"


def _trust_models(lots_) -> dict:
    """Per READ_SETS name: the four quarters' trust weights fitted on the masked 2000 prints, their held-out log loss,
    and the street-side rates of the same readings."""
    from sfzmaps.solver import streetside, trust
    prints = trust.load_prints()
    sides = {q: np.load(paths.LABELS_DIR / f"streetside_{q}.npz") for q in paths.QUARTERS}
    out = {}
    for name, reads in READ_SETS.items():
        p, kept = mask_reads(prints, reads)
        table = trust.features(p, lots_)
        weights, rates, loss = [], [], {}
        for qi, q in enumerate(paths.QUARTERS):
            w = trust.fit(table, q)
            chance = trust.trust(table, w)
            held = table.known & (table.quarter == qi)
            loss[q] = round(trust.log_loss(chance[held], table.target[held]), 4)
            top = streetside.top_codes(table.names, table.print_index, table.code_index, chance, len(p))
            S = sides[q]
            r = streetside.rates(top, p["lab_lot"].to_numpy(np.int64), S["lotB"][kept], S["onst"][kept],
                                 S["isfw"][kept], lots_.codes, lots_.known, lots_.quarter, q)
            weights.append(w)
            rates.append([r[k] for k in ("rA_st", "rB_st", "rA_fw", "rB_fw")])
        out[name] = {"weights": np.array(weights), "rates": np.array(rates), "held_out_log_loss": loss,
                     "prints": len(p), "readings": len(table.print_index)}
        log.info("trust %s: %d prints, held-out log loss %s", name, len(p), loss)
    return out


def build_models(workers: int = 2) -> dict:
    """Fit the 2000 map's models for every other series (see the module) and write them into paths.TRANSFER_DIR:
    walls.npz, trust_<read set>.npz, lotsize.npz, placement.npz and models.json.  Runs on the 2000 map's stages'
    outputs (its lots, prints, labels, walls and text chance)."""
    from sfzmaps.solver import lots, placement, walls
    if not paths.is_2000():
        raise ValueError("the transfer models are fitted on the 2000 map: run without --series")
    t0 = time.perf_counter()
    out = paths.TRANSFER_DIR
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    lots_ = lots.load_lots()
    for name, m in _trust_models(lots_).items():
        np.savez_compressed(out / f"trust_{name}.npz", weights=m["weights"], rates=m["rates"])
        summary[f"trust_{name}"] = {k: v for k, v in m.items() if k not in ("weights", "rates")}
    shutil.copyfile(paths.LABELS_DIR / "lotsize.npz", out / "lotsize.npz")
    rows = placement._options()
    models = [placement.fit_model(rows, qi) for qi in range(4)]
    np.savez_compressed(out / "placement.npz", mu=np.array([m[0] for m in models]),
                        sd=np.array([m[1] for m in models]), w=np.array([m[2] for m in models]))
    summary["placement"] = {"labels": len(rows)}
    log.info("placement models fitted (%.0f s)", time.perf_counter() - t0)
    inputs = walls.load_inputs([list(TRANSFER_WALLS)])
    X = walls.candidate_design(inputs, list(TRANSFER_WALLS), workers)
    y = walls.table_lines(inputs.lots)
    W, quarters = [], {}
    for qi, q in enumerate(paths.QUARTERS):
        train, held = walls.training_pairs(inputs.lots, qi), walls.held_out_pairs(inputs.lots, qi)
        w = walls.fit_quarter(X, y, train)
        lo = X @ w
        quarters[q] = {"held_out_log_loss": round(walls.log_loss(lo[held], y[held]), 6),
                       "held_out_pairs": int(held.sum())}
        W.append(w)
    np.savez_compressed(out / "walls.npz", weights=np.array(W), candidates=np.array(TRANSFER_WALLS))
    summary["walls"] = {"candidates": list(TRANSFER_WALLS), "features": int(X.shape[1]), "quarters": quarters}
    # No run time in the file: a stage's output must be the same every run, or every series downstream goes stale
    (out / "models.json").write_text(json.dumps(summary, indent=1) + "\n")
    seconds = round(time.perf_counter() - t0, 1)
    log.info("transfer models written (%.0f s)", seconds)
    return {**summary, "seconds": seconds}


def _models(name: str):
    f = paths.TRANSFER_DIR / name
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver transfer-models` (on the 2000 map)")
    return np.load(f)


def placement_models() -> list:
    """The 2000 map's four placement models, (mu, sd, w) each, nw ne sw se."""
    P = _models("placement.npz")
    return [(P["mu"][qi], P["sd"][qi], P["w"][qi]) for qi in range(len(paths.QUARTERS))]


def build_labels() -> dict:
    """A series' labels from the 2000 map's models (the `labels` stage of a series other than 2000): each reading's
    trust (readings_{q}.npz, trust.json), the street sides (the series' far-side lots, the 2000 rates), the lot-size
    evidence (2000's) and the strokes of the series' own ink, into LABELS_DIR."""
    from sfzmaps.solver import lots, strokes, streetside, trust
    import geopandas as gpd
    t0 = time.perf_counter()
    prints = trust.load_prints()
    lots_ = lots.load_lots()
    name = read_set(prints)
    T = _models(f"trust_{name}.npz")
    table = trust.features(prints, lots_)
    paths.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"read_set": name, "prints": len(prints), "readings": len(table.print_index),
               "names": len(table.names), "quarters": {}}
    for qi, q in enumerate(paths.QUARTERS):
        chance = trust.trust(table, T["weights"][qi])
        np.savez_compressed(paths.LABELS_DIR / f"readings_{q}.npz", names=np.array(table.names),
                            print_index=table.print_index, code_index=table.code_index, trust=chance)
        held = table.known & (table.quarter == qi)
        summary["quarters"][q] = {"held_out": int(held.sum()),
                                  "held_out_log_loss": round(trust.log_loss(chance[held], table.target[held]), 4)}
    (paths.LABELS_DIR / "trust.json").write_text(json.dumps(summary, indent=1) + "\n")
    xy = prints[["x", "y"]].to_numpy(float)
    under = prints["lab_lot"].to_numpy(np.int64)
    street = prints["street"].to_numpy(bool)
    geoms = gpd.read_parquet(paths.LOTS_DIR / "table.parquet").geometry.values
    streets = streetside.load_streets()
    far, nearest = streetside.far_side(xy, under, geoms, streets.geometry.values)
    freeway = streetside.freeways(streets)[nearest] if len(nearest) else np.zeros(0, bool)
    for qi, q in enumerate(paths.QUARTERS):
        rates = dict(zip(("rA_st", "rB_st", "rA_fw", "rB_fw"), map(float, T["rates"][qi])))
        np.savez_compressed(paths.LABELS_DIR / f"streetside_{q}.npz", lotB=far, onst=street, isfw=freeway, **rates)
    shutil.copyfile(paths.TRANSFER_DIR / "lotsize.npz", paths.LABELS_DIR / "lotsize.npz")
    s = strokes.strokes(__import__("sfzmaps.solver.page", fromlist=["load_ink"]).load_ink())
    np.savez_compressed(paths.LABELS_DIR / "strokes.npz", **s)
    summary["strokes"] = int(len(s["length"]))
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    return summary


def build_walls(workers: int = 2) -> dict:
    """A series' wall odds from the 2000 map's weights (the `walls` stage of a series other than 2000): the series'
    border features under TRANSFER_WALLS, each quarter's odds by the weights 2000 fitted without that quarter, into
    WALLS_DIR (walls_{q}.npz, evaluation.json: on a series with the table, the held-out log loss against it), with
    the text chance it read (text_chance.png)."""
    from sfzmaps.solver import runs, textgrid, walls
    t0 = time.perf_counter()
    inputs = walls.load_inputs([list(TRANSFER_WALLS)])
    scale = width_scale(inputs.strokes)
    if scale != 1.0:                                 # the drawing's pen widths on the 2000 drawing's scale
        for k in ("w", "wmed", "wiqr"):
            inputs.strokes[k] = (inputs.strokes[k].astype(np.float64) * scale).astype(inputs.strokes[k].dtype)
    X = walls.candidate_design(inputs, list(TRANSFER_WALLS), workers)
    W = _models("walls.npz")
    if tuple(W["candidates"].astype(str)) != TRANSFER_WALLS:
        raise ValueError("the 2000 walls model was fitted with other candidates: rerun `sfzmaps solver transfer-models`")
    if X.shape[1] != W["weights"].shape[1]:
        raise ValueError(f"{X.shape[1]} wall features, the 2000 model has {W['weights'].shape[1]}")
    y = walls.table_lines(inputs.lots)
    odds, quarters = {}, {}
    for qi, q in enumerate(paths.QUARTERS):
        lo = X @ W["weights"][qi]
        odds[q] = lo.astype(np.float32)
        held = walls.held_out_pairs(inputs.lots, qi)
        quarters[q] = {"held_out_log_loss": round(walls.log_loss(lo[held], y[held]), 6) if held.any() else None,
                       "held_out_pairs": int(held.sum())}
    evaluation = {"pairs": len(y), "features": int(X.shape[1]), "quarters": quarters,
                  "candidates": list(TRANSFER_WALLS), "weights": "2000 (transfer)", "width_scale": scale}
    folder = runs.fresh_folder(runs.WALLS_DIR)
    walls.write_walls(folder, odds, evaluation)
    textgrid.write_text_chance(folder, inputs.text)
    runs.replace_folder(folder, runs.WALLS_DIR)
    return {**evaluation, "seconds": round(time.perf_counter() - t0, 1)}


WIDTH_SCALE_ENV = "SFZMAPS_WALL_WIDTH_SCALE"


def width_scale(strokes) -> float:
    """How much wider the 2000 drawing's pen is than the series' (WIDTH_SCALE_ENV: "auto" for the ratio of the two
    pages' length-weighted mean stroke widths, measured on the ink alone; a number for that; "1" for none).  Unset:
    none for a scan of the 2000 drawing (paths.FRAME_SERIES: the same pen), "auto" for any other drawing.  The 2000
    walls weigh stroke widths in feet, and the older drawings print every line thinner.  Measured: on 1960 the share
    of its own labels the solve keeps rose from 41% to 82% (scale 1.32); on the 1996-2000 scan (scale 1.05) the
    table's wrong lots rose from 15,974 to 20,169, so a scan of the 2000 drawing is not scaled."""
    v = os.environ.get(WIDTH_SCALE_ENV) or ("1" if paths.SERIES in paths.FRAME_SERIES else "auto")
    if v != "auto":
        return float(v)
    ref = np.load(paths.solver_dir(paths.SERIES_2000) / "labels" / "strokes.npz")

    def mean(s):
        w, L = s["wmed"].astype(float), s["length"].astype(float)
        return float((w * L).sum() / L.sum())
    return round(mean(ref) / mean(strokes), 4)


def build_text_chance(device: str | None = None) -> dict:
    """The one detector's (2000's `solver-text` model, paths.TEXT_MODEL) chance of text on every sheet of the series,
    into TEXT_CHANCE_DIR/<sheet>.png, scored in one exclusive turn on the GPU."""
    import cv2
    import torch

    from sfzmaps import gpu
    from sfzmaps.solver import grid
    from sfzmaps.solver.textnet import data, predict, train
    device = device or train.default_device()
    t0 = time.perf_counter()
    paths.TEXT_CHANCE_DIR.mkdir(parents=True, exist_ok=True)
    with gpu.gpu_turn(True, device):
        net = predict.load_model(paths.TEXT_MODEL, device)
        for name in paths.SHEETS:
            ink = data.load_page(name) < grid.INK_BELOW
            if not cv2.imwrite(str(paths.TEXT_CHANCE_DIR / f"{name}.png"), predict.chance(net, ink, device)):
                raise OSError(f"can't write {paths.TEXT_CHANCE_DIR / f'{name}.png'}")
            log.info("%s scored (%.0f s)", name, time.perf_counter() - t0)
        del net
        if gpu.on_gpu(device):
            torch.mps.empty_cache()
    return {"sheets": len(paths.SHEETS), "device": device, "seconds": round(time.perf_counter() - t0, 1)}


def era_canon(code: str) -> str:
    """A code in the one spelling of the series' era: on a scan of the 2000 drawing (paths.FRAME_SERIES) or a height
    map, the 2000 map's (`codes.canon`, which keeps each named Neighborhood Commercial District's name); on an older
    drawing, the pipeline's era-aware spelling (`districts.rules.normalise_code` in the series' year: 'C 1' and 'C1'
    are C-1) where it knows the code, else `codes.canon`.  (normalise_code turns every named NCD into "NCD", which
    on 1996-2000 dropped every named district's label from the solve: 20,613 wrong lots on the table, not 15,974.)"""
    from sfzmaps.districts.rules import normalise_code
    from sfzmaps.districts.series import SERIES
    from sfzmaps.solver.codes import canon
    if paths.SERIES.endswith("-height") or paths.SERIES in paths.FRAME_SERIES:
        return canon(code)
    return normalise_code(code, SERIES[paths.SERIES].year) or canon(code)


def series_vocabulary() -> set[str]:
    """The codes a series' labels may name (the solve drops readings of any other): the codes of its published map
    (the pipeline's reader, held to the era's legend and code list) with every code read on its prints that the era's
    research list knows (`districts.rules.rule_for` in the series' year); on a height map, the period's published
    heights and Sol's reads of that form.  Never the April 2000 table's codes."""
    import pandas as pd

    from sfzmaps import config
    from sfzmaps.districts.rules import rule_for
    from sfzmaps.districts.series import SERIES
    from sfzmaps.solver.codes import canon
    from sfzmaps.solver.trust import READS, load_prints
    if paths.SERIES.endswith("-height"):             # a height map: the period's published heights, and Sol's reads
        period = SERIES[paths.SERIES].period          # of the same form (a number, a dash and a letter; OS)
        import re
        heights = pd.read_parquet(config.OUT_DIR / f"{period}-zoning" / "districts" / "height.parquet",
                                  columns=["height_code"])["height_code"].dropna().astype(str)
        out = {canon(c) for c in heights if c}
        prints = load_prints()
        out |= {canon(c) for c in set(prints["sol_code"].astype(str)) - {""}
                if re.fullmatch(r"\d+(-[A-Z])?|OS", canon(c))}
        return out
    year = SERIES[paths.SERIES].year
    out = set()
    f = config.WORK_DIR / paths.SERIES / "districts_raw.parquet"
    if f.is_file():
        out |= {era_canon(c) for c in pd.read_parquet(f, columns=["code"])["code"].dropna().astype(str) if c}
    prints = load_prints()
    for col in READS:
        for c in set(prints[col].astype(str)) - {""}:
            if rule_for(c, year) is not None:
                out.add(era_canon(c))
    return out
