"""Issue #71 audit: is the first look's big error (1948/14's Lake Merced cut, removed-block false alarms) from fold 0
holding out the only sheets of their kind?  Train the same "fold 16" net, seed 0, 4000 steps, on ALL labelled sheets
and predict the same unlabelled pages (every 1948 page + the close-up pages) with the per-pixel cut at the fold-0
temperature.  Throwaway.

    uv run python scripts/spikes/issue71_all_labels.py
"""
import json
import logging

from sfzmaps import gpu
from sfzmaps.frame import data, learn, look
from sfzmaps.frame import run as fr

OUT = data.DATA_DIR / "frame-all-labels"
T = float(json.loads((data.DATA_DIR / "frame-arch-fold16" / "fold-0" / fr.FOLDS_FILE).read_text())["temperature"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pages = list(fr.labelled_pages(data.labelled_sheets()).values())
    labelled = {p.id for p in pages}
    sids = sorted(({s for s in data.map_pages(data.PAGE_KINDS, data.WORK) if s[0] == "1948"}
                   | {(s, h) for s, h, _, _ in look.CLOSEUPS}) - labelled)
    with gpu.gpu_turn(True, "mps"):
        net, _ = learn.train(pages, learn.Recipe(), "mps", snapshots=OUT / "snap", label="all labels")
        fr.predict_pages([net], T, sids, "mps", OUT)
    look.series_mosaics(OUT)
    look.closeups(OUT)


if __name__ == "__main__":
    main()
