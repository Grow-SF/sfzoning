"""The prints: every zoning code printed on the 2000 sheets, where it is, and every saved read of it.

Two readers found the printed codes on each sheet's page: Terra's text boxes (the saved answers of
`text-on-every-sheet`, decoded as the script that asked them decoded them) and Sol's label points
(`district_labels.json`, kind "use").  A Terra zoning-code box with Sol points inside it is one print read by both
(kind 0): the point nearest the box's centre gives Sol's read, and every point inside the box is used up.  A box with
no point is Terra's alone (kind 1); a point inside no box is Sol's alone (kind 2).  A print sits at its box's centre
(or Sol's point) on the page and on the map through the sheet's corrected placement; it is kept when that point is
inside the sheet's warped footprint and on the mapped area (`sheets.png` > 0).

Three more saved reads join the prints:
  - the turned pass (`sideways-text`: every tile shown a quarter turned, its boxes in page px): a turned zoning-code
    box is one more reading of every print of its sheet whose point is inside it (edges included) or whose Terra box
    it overlaps by IoU > TURNED_JOIN_IOU; of several, the box farthest from its tile's edge wins.  A turned box on no
    print whose code is a known one (the table's, or one Sol reads on a print) is a print of its own (kind 3), kept as
    above.
  - the third read (`third-code-read`): a close crop of the page around each print of kinds 0-2, asking which code is
    printed there; and its re-check (`third-code-read-recheck`): the crop turned a quarter each way, for codes printed
    taller than wide.  These answers name no sheet or picture, so each print's crop and question are made again
    exactly as they were asked and the answer is found by its cache key.  A key not found is counted and that print
    has no such read; with fewer than MIN_KEY_MATCH of the keys found, the stage stops.  The key also names the model
    asked; it is made with the model, picture limit and point-box size the third reads were asked with
    (THIRD_READ_MODEL_PARTS, THIRD_READ_MAX_EDGE, POINT_CODE_SIZE), never today's settings.

Writes, under data/solver/prints/:
  prints.parquet    one row per print, in the order: sheet by sheet, Terra's boxes then Sol's other points; then the
                    prints of the turned pass, sheet by sheet.  Columns:
                      sheet, page_x, page_y      the sheet, and the print's point on its page (px)
                      x, y                       that point on the map, EPSG:2227 ft
                      kind                       0 Terra and Sol, 1 Terra only, 2 Sol only, 3 the turned pass only
                      box_x0 .. box_y1           Terra's box, page px (NaN without one)
                      terra_code                 Terra's read, in the one spelling (`codes.canon`; "" for none)
                      sol_code, sol_confidence   Sol's read and how sure it was
                      turned_code, turned_edge   the turned pass's read, and its box's distance from its tile's edge
                      third_code, third_confidence, third_leader, third_found
                                                 the third read: the code, how sure, a leader line seen, and whether a
                                                 saved answer was found
                      recheck_asked, recheck_code, recheck_confidence, recheck_found
                                                 the turned re-check: asked of this print, the surer of its two reads
                      lab_lot                    the lot under the point (`grown`: in a street, the nearest lot)
                      street                     the point is in no lot (`lot` < 0)
  terra_boxes.json  every sheet's decoded Terra boxes, every kind, in page px (what the text detector learns from)
  summary.json      prints by kind, and the saved answers asked, found and missing for each read
"""
import json
import math
import time

import cv2
import numpy as np
import pandas as pd
import shapely
from pydantic import BaseModel

from sfzmaps import config, llm
from sfzmaps.solver import codes, grid, lots, page, paths, saved
from sfzmaps.solver.textnet import boxes as textboxes
from sfzmaps.transform import SheetTransform

TERRA_READ = textboxes.TERRA_READ
TURNED_POSITIONS = "experiments-2000/text-positions/sideways-text"
THIRD_READ = "experiments-2000/third-code-read"
RECHECK_READ = "experiments-2000/third-code-read-recheck"

TURNED_JOIN_IOU = 0.3     # a turned box overlapping a print's Terra box by more is one more reading of that print
CROP_REACH = 1.5          # the third read's crop reaches this many code sizes from the code's centre each way
CROP_MIN_EDGE = 640       # ... and is enlarged by a whole factor until its long edge is at least this (px)
MIN_KEY_MATCH = 0.95      # fewer of a read's saved answers found than this share of those asked: stop
# How the third reads were asked (a historical fact, not a setting): of the pipeline's default vision model then, Sol
# at low effort, whose key has the first format's parts (no trailing parameters), with pictures up to 2048 px.  The
# keys are made with these, whatever the pipeline's model is today.
THIRD_READ_MODEL_PARTS = (["openai", "gpt-6.1-sol"], [])
THIRD_READ_MAX_EDGE = 2048
# The box the third read was asked about around a bare Sol point: this wide and half as high (px).  It is read3.py's
# `med`, the median long side of the Terra boxes of the prints it read (the first run's 1,771 with a box), measured
# on today's kept prints (64.0 from 1,769 boxes); every saved key of a point print is found with it.
POINT_CODE_SIZE = 64.0


# The answer format, copied from the scripts that asked (Terra's is `textnet.boxes.TextItems`).  A docstring would
# enter the JSON schema (as its "description") and so the cache key: this class keeps the scripts' name and fields,
# and no docstring.
class CodeRead(BaseModel):    # the third read's answer (third-code-read and its re-check)
    code: str
    confidence: float
    leader: bool


THIRD_PROMPT = """This image is a close crop of a scanned San Francisco zoning district map. It is {w} pixels wide and {h} pixels high. A zoning district code is printed in the middle of the image, within about the rectangle from ({x0}, {y0}) to ({x1}, {y1}) (pixels, origin at the top-left). Read that one code exactly as printed, character by character; take particular care with the digit after the hyphen and with any suffix in brackets. Codes look like RH-1, RH-1(D), RH-1(S), RH-2, RH-3, RM-1, RM-2, RM-3, RM-4, RC-3, RC-4, NC-1, NC-2, NC-3, NC-S, C-1, C-2, C-3-O, C-3-R, C-3-G, C-3-S, C-M, M-1, M-2, P, RED, SLR, SLI, SSO, RSD, SPD; some districts are printed as a name instead (for example NORTH BEACH). Street names, block numbers and other lettering are not zoning codes. Return: code = the code as printed (an empty string if what is printed there is not a zoning district code); confidence = from 0 to 1, how sure you are of every character; leader = true only if a drawn leader line or arrow runs from this code to an area other than where the code itself is printed."""  # noqa: E501
TURNED_NOTE = " The crop has been turned a quarter turn; if the code now reads upside down, still read it."

COLUMNS = ["sheet", "page_x", "page_y", "x", "y", "kind", "box_x0", "box_y0", "box_x1", "box_y1", "terra_code",
           "sol_code", "sol_confidence", "turned_code", "turned_edge", "third_code", "third_confidence",
           "third_leader", "third_found", "recheck_asked", "recheck_code", "recheck_confidence", "recheck_found",
           "lab_lot", "street"]


def terra_boxes(sheet: str, records: list[dict] | None = None) -> list[dict]:
    """A 2000 sheet's Terra text boxes, every kind, in page px, decoded as `textnet.boxes.terra_boxes` decodes any
    sheet's.  `records`: the lines of `text-on-every-sheet` (read from disk when None)."""
    return textboxes.terra_boxes(paths.SERIES, sheet, records)


def _code(text: str) -> str:
    """A read in the one spelling: upper case, single spaces, `codes.canon` ("" stays "")."""
    t = " ".join(str(text).upper().split())
    return codes.canon(t) if t else ""


def _confidence(label: dict) -> float:
    c = label.get("confidence")
    return math.nan if c is None else float(c)


def _row(page_x, page_y, kind, box=None, terra_code="", sol_code="", sol_confidence=math.nan, turned_code="",
         turned_edge=-math.inf) -> dict:
    """A print found on a page, its other reads still empty."""
    return {"page_x": float(page_x), "page_y": float(page_y), "kind": kind, "box": box, "terra_code": terra_code,
            "sol_code": sol_code, "sol_confidence": sol_confidence, "turned_code": turned_code,
            "turned_edge": turned_edge, "third_code": "", "third_confidence": math.nan, "third_leader": False,
            "third_found": False, "recheck_asked": False, "recheck_code": "", "recheck_confidence": math.nan,
            "recheck_found": False}


def label_kind() -> str:
    """The kind of Sol's label points that are the series' prints: "height" on a height map (`<period>-height`),
    "use" (zoning) on every other."""
    return "height" if paths.SERIES.endswith("-height") else "use"


def match_prints(boxes: list[dict], labels: list[dict]) -> list[dict]:
    """One sheet's prints in page px from its Terra boxes (`terra_boxes`) and Sol's labels (district_labels.json's
    "labels"): each zoning-code box, with the read of the Sol point inside it nearest its centre (kind 0) or none
    (kind 1), then each of Sol's "use" points (`label_kind`: "height" on a height map) inside no zoning-code box
    (kind 2).  Points on a box's edge are inside it."""
    kind = label_kind()
    use = [lab for lab in labels if lab.get("kind", "use") == kind]
    pts = np.array([[lab["x"], lab["y"]] for lab in use], float).reshape(-1, 2)
    used = np.zeros(len(use), bool)
    rows = []
    for b in boxes:
        if b["kind"] != "zoning code":
            continue
        bx0, by0, bx1, by1 = b["box"]
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        sol, confidence = "", math.nan
        inside = np.flatnonzero((pts[:, 0] >= bx0) & (pts[:, 0] <= bx1) & (pts[:, 1] >= by0) & (pts[:, 1] <= by1))
        if len(inside):
            j = inside[np.argmin(np.hypot(pts[inside, 0] - cx, pts[inside, 1] - cy))]
            used[inside] = True
            sol, confidence = use[j]["code"].upper().strip(), _confidence(use[j])
        rows.append(_row(cx, cy, 0 if sol else 1, box=(bx0, by0, bx1, by1), terra_code=_code(b["text"]),
                         sol_code=codes.canon(sol) if sol else "", sol_confidence=confidence))
    for j in np.flatnonzero(~used):
        rows.append(_row(pts[j, 0], pts[j, 1], 2, sol_code=codes.canon(use[j]["code"].upper().strip()),
                         sol_confidence=_confidence(use[j])))
    return rows


def join_turned(prints: list[dict], turned: list[dict], codes_known: set[str]) -> list[dict]:
    """Join one sheet's turned-pass boxes (page px, in their file's order) to its prints (changed in place): a
    zoning-code box holding a print's point (its edges included, as when matching Terra's boxes to Sol's points), or
    overlapping the print's Terra box by IoU > TURNED_JOIN_IOU, is a reading of that print, and the box with the
    largest `edge` wins.  Returns the new prints (kind 3, at the box's centre): boxes joining no print whose code is
    in `codes_known`."""
    new = []
    for b in turned:
        if b["kind"] != "zoning code":
            continue
        code = _code(b["text"])
        x0, y0, x1, y1 = b["box"]
        hit = [p for p in prints if (x0 <= p["page_x"] <= x1 and y0 <= p["page_y"] <= y1)
               or (p["box"] is not None and textboxes.iou(b["box"], p["box"]) > TURNED_JOIN_IOU)]
        for p in hit:
            if b["edge"] > p["turned_edge"]:
                p["turned_code"], p["turned_edge"] = code, b["edge"]
        if not hit and code in codes_known:
            new.append(_row((x0 + x1) / 2, (y0 + y1) / 2, 3, turned_code=code, turned_edge=b["edge"]))
    return new


def code_box(p: dict, size: float) -> tuple[tuple, float]:
    """The page box the third read asks about, and the code's size: Terra's box and its long side; for a point (or
    an empty box), a box `size` wide and `size` / 2 high around it."""
    bx0, by0, bx1, by1 = p["box"] if p["box"] is not None else (p["page_x"], p["page_y"], p["page_x"], p["page_y"])
    sz = max(bx1 - bx0, by1 - by0)
    if sz <= 0:
        return (bx0 - size / 2, by0 - size / 4, bx1 + size / 2, by1 + size / 4), size
    return (bx0, by0, bx1, by1), sz


def third_read_crop(page_gray: np.ndarray, box, size: float) -> tuple[np.ndarray, tuple]:
    """The close crop the third read was shown: the page within CROP_REACH sizes of the box's centre, enlarged by a
    whole factor (cubic) until its long edge is at least CROP_MIN_EDGE (never past THIRD_READ_MAX_EDGE).  Returns
    the crop and the box in the crop's px (floats)."""
    bx0, by0, bx1, by1 = box
    cx, cy, half = (bx0 + bx1) / 2, (by0 + by1) / 2, size * CROP_REACH
    r0, r1 = int(max(0, cy - half)), int(min(page_gray.shape[0], cy + half))
    c0, c1 = int(max(0, cx - half)), int(min(page_gray.shape[1], cx + half))
    crop = page_gray[r0:r1, c0:c1]
    z = max(1, int(np.ceil(CROP_MIN_EDGE / max(crop.shape))))
    z = min(z, max(1, THIRD_READ_MAX_EDGE // max(crop.shape)))
    if z > 1:
        crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    return crop, ((bx0 - c0) * z, (by0 - r0) * z, (bx1 - c0) * z, (by1 - r0) * z)


def third_read_prompt(crop: np.ndarray, box) -> str:
    """The third read's question about a crop, the code's box in its px."""
    return THIRD_PROMPT.format(w=crop.shape[1], h=crop.shape[0], x0=int(box[0]), y0=int(box[1]), x1=int(box[2]),
                               y1=int(box[3]))


def recheck_crops(crop: np.ndarray, box) -> list[tuple[np.ndarray, str]]:
    """The re-check's two pictures and questions: the crop turned a quarter clockwise, then anticlockwise, the code's
    box turned with it."""
    h, w = crop.shape[:2]
    x0, y0, x1, y1 = box
    out = []
    for rot, turn in ((cv2.ROTATE_90_CLOCKWISE, lambda x, y: (h - 1 - y, x)),
                      (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda x, y: (y, w - 1 - x))):
        turned = cv2.rotate(crop, rot)
        (xa, ya), (xb, yb) = turn(x0, y0), turn(x1, y1)
        out.append((turned, third_read_prompt(turned, (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)))
                    + TURNED_NOTE))
    return out


def _key(crop: np.ndarray, prompt: str) -> str:
    """The saved cache key of a third-read question: asked of THIRD_READ_MODEL_PARTS, not today's model."""
    png = llm._encode(crop, max_edge=THIRD_READ_MAX_EDGE)
    return saved.cache_key(llm._key(png, prompt, CodeRead, model_parts=THIRD_READ_MODEL_PARTS))


def third_reads(prints: list[dict], pages: dict[str, np.ndarray], third: dict[str, dict],
                recheck: dict[str, dict]) -> dict:
    """Find the third read and its re-check of every print of kinds 0-2 (changed in place) in the saved answers
    `third` and `recheck` (by cache key, `saved.by_key`), each crop and question made again as they were asked; a
    point is asked about with a box of POINT_CODE_SIZE.  The re-check is asked of codes printed taller than wide whose
    longest read has more than one letter; of its two answers the surer one with a code wins.  `pages`: each sheet's
    page, grayscale.  Returns, per read, the keys asked, found and missing, the saved answers, and those no print
    asked for."""
    asked = [p for p in prints if p["kind"] < 3]
    found = {"third_read": set(), "recheck": set()}
    counts = {"third_read": {"asked": 0, "found": 0, "missing": 0}, "recheck": {"asked": 0, "found": 0, "missing": 0}}

    def look(read, table, key):
        counts[read]["asked"] += 1
        rec = table.get(key)
        counts[read]["found" if rec is not None else "missing"] += 1
        if rec is None:
            return None
        found[read].add(key)
        return CodeRead.model_validate(rec["answer"])

    for p in asked:
        box, sz = code_box(p, POINT_CODE_SIZE)
        crop, cbox = third_read_crop(pages[p["sheet"]], box, sz)
        p.update(third_code="", third_confidence=math.nan, third_leader=False, third_found=False,
                 recheck_code="", recheck_confidence=math.nan, recheck_found=False,
                 recheck_asked=(box[3] - box[1]) > (box[2] - box[0])
                 and max(len(p["sol_code"]), len(p["terra_code"])) > 1)
        a = look("third_read", third, _key(crop, third_read_prompt(crop, cbox)))
        if a is not None:
            p.update(third_code=_code(a.code), third_confidence=a.confidence, third_leader=a.leader,
                     third_found=True)
        if not p["recheck_asked"]:
            continue
        best, surest = "", -1.0
        for turned, prompt in recheck_crops(crop, cbox):
            a = look("recheck", recheck, _key(turned, prompt))
            if a is None:
                continue
            p["recheck_found"] = True
            if _code(a.code) and a.confidence > surest:
                best, surest = _code(a.code), a.confidence
        if best:
            p["recheck_code"], p["recheck_confidence"] = best, surest
    for read, table in (("third_read", third), ("recheck", recheck)):
        counts[read].update(saved=len(table), unused=len(table) - len(found[read]))
    return counts


class SavedReadsMissing(RuntimeError):
    """Too few of a read's saved answers were found by their rebuilt cache keys."""


def check_key_match(counts: dict) -> None:
    """Stop (SavedReadsMissing) when, for any read, fewer than MIN_KEY_MATCH of the keys asked were found."""
    short = [f"{read}: {c['found']} of {c['asked']} saved answers found ({c['found'] / c['asked']:.1%})"
             for read, c in counts.items() if c["asked"] and c["found"] < MIN_KEY_MATCH * c["asked"]]
    if short:
        raise SavedReadsMissing("; ".join(short) + f"; under {MIN_KEY_MATCH:.0%}: the crops, questions or vision "
                                "model differ from those asked")


def under(xy: np.ndarray, lot: np.ndarray, grown: np.ndarray, mapped: np.ndarray):
    """For map points (n, 2): on the mapped area (bool), the lot under each (`grown`, int32; -1 where not on it) and
    in a street (`lot` < 0).  A point is read at the grid px holding it, floor(to_grid)."""
    cr = np.floor(grid.to_grid(np.asarray(xy, float).reshape(-1, 2))).astype(np.int64)
    c, r = cr[:, 0], cr[:, 1]
    h, w = mapped.shape
    on = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    on[on] = mapped[r[on], c[on]]
    lab = np.full(len(cr), -1, np.int32)
    street = np.zeros(len(cr), bool)
    lab[on] = grown[r[on], c[on]]
    street[on] = lot[r[on], c[on]] < 0
    return on, lab, street


def _place(found: list[dict], sheet: str, transform, footprint, lot_rasters, mapped) -> list[dict]:
    """The prints of `found` whose point lands inside the sheet's footprint and on the mapped area, with the sheet,
    their map point, the lot under it and whether it is in a street."""
    if not found:
        return []
    xy = transform.to_world(np.array([[p["page_x"], p["page_y"]] for p in found], float))
    inside = shapely.contains(footprint, shapely.points(xy))
    on, lab, street = under(xy, lot_rasters.lot, lot_rasters.grown, mapped)
    return [{**p, "sheet": sheet, "x": float(x), "y": float(y), "lab_lot": int(lb), "street": bool(st)}
            for p, (x, y), keep, lb, st in zip(found, xy, inside & on, lab, street) if keep]


def _frame(rows: list[dict]) -> pd.DataFrame:
    """The prints as prints.parquet's table."""
    out = []
    for p in rows:
        box = p["box"] if p["box"] is not None else (math.nan,) * 4
        out.append({**{c: p.get(c) for c in COLUMNS}, "box_x0": box[0], "box_y0": box[1], "box_x1": box[2],
                    "box_y1": box[3], "turned_edge": p["turned_edge"] if p["turned_code"] else math.nan})
    df = pd.DataFrame(out, columns=COLUMNS)
    for c in ("page_x", "page_y", "x", "y", "box_x0", "box_y0", "box_x1", "box_y1", "sol_confidence", "turned_edge",
              "third_confidence", "recheck_confidence"):
        df[c] = df[c].astype("float64")
    for c in ("third_leader", "third_found", "recheck_asked", "recheck_found", "street"):
        df[c] = df[c].astype(bool)
    return df.astype({"kind": "int8", "lab_lot": "int32"})


def build_prints() -> dict:
    """Find every print of the 2000 sheets with its reads and write prints.parquet, terra_boxes.json and
    summary.json into PRINTS_DIR.  Returns the summary and the seconds taken."""
    t0 = time.perf_counter()
    names = paths.sheet_names()
    if not names:
        raise FileNotFoundError(f"no warped sheets under {config.DATA_DIR / 'warp' / paths.SERIES}: "
                                f"run the warp-{paths.SERIES} stage first")
    terra = saved.answers(TERRA_READ)
    # The turned pass and the third reads were asked of the 2000 sheets alone; another series' prints are Terra's
    # boxes and Sol's points, where they read its sheets (a sheet neither read has no prints).
    turned = saved.answers(TURNED_POSITIONS) if paths.is_2000() else []
    lot_rasters = lots.load_lots()
    mapped = page.load_sheets() > 0
    boxes, placed, rows = {}, {}, []
    for name in names:
        warp = config.DATA_DIR / "warp" / paths.SERIES / name
        transform = SheetTransform.from_json(json.loads((warp / "transform.json").read_text()), folder=warp)
        footprint = shapely.from_geojson((warp / "warped.footprint.geojson").read_text())
        boxes[name] = terra_boxes(name, terra)
        placed[name] = transform, footprint
        has_sol = paths.is_2000() or (config.WORK_DIR / paths.SERIES / name / "district_labels.json").is_file()
        rows += _place(match_prints(boxes[name], saved.sol_labels(name) if has_sol else []), name, transform,
                       footprint, lot_rasters, mapped)
    codes_known = set(lot_rasters.codes[lot_rasters.known]) | {p["sol_code"] for p in rows if p["sol_code"]}
    new = []
    for name in names:
        found = join_turned([p for p in rows if p["sheet"] == name], [b for b in turned if b["sheet"] == name],
                            codes_known)
        new += _place(found, name, *placed[name], lot_rasters, mapped)
    counts = {}
    if paths.is_2000():
        pages = {}
        for name in names:
            page_file = config.WORK_DIR / paths.SERIES / name / "page.png"
            pages[name] = cv2.imread(str(page_file), cv2.IMREAD_GRAYSCALE)
            if pages[name] is None:
                raise FileNotFoundError(f"can't read {page_file}")
        counts = third_reads(rows, pages, saved.by_key(THIRD_READ), saved.by_key(RECHECK_READ))
        check_key_match(counts)
    df = _frame(rows + new)
    summary = {"prints": len(df), "by_kind": {str(k): int((df["kind"] == k).sum()) for k in range(4)},
               "saved_answers": counts}
    paths.PRINTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.PRINTS_DIR / "prints.parquet", index=False)
    (paths.PRINTS_DIR / "terra_boxes.json").write_text(json.dumps(boxes) + "\n")
    (paths.PRINTS_DIR / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return {**summary, "seconds": round(time.perf_counter() - t0, 1)}


def load_prints() -> pd.DataFrame:
    """Read prints.parquet."""
    f = paths.PRINTS_DIR / "prints.parquet"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver prints`")
    return pd.read_parquet(f)
