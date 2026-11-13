"""The pipeline's district label reads (Sol), every sheet of the given series (default: every series that has them),
asked again through the pipeline's own reader with nothing written (districts/labels.py _read, write=False), so a
recorded run (python -m saved_reads run ... pipeline_labels.py) logs every tile's question.  Each sheet is read with
the settings its district_labels.json was made with (scale, tile, overlap: the read key stored with it); a sheet whose
stored key no longer matches today's prompt is read with today's, and its old answers show up as misses.
usage: pipeline_labels.py [SERIES ...]"""
import sys

from sfzmaps import config, llm
from sfzmaps.districts import labels as L

series = sys.argv[1:] or sorted(p.name for p in config.WORK_DIR.iterdir()
                                if p.is_dir() and any(p.glob("*/district_labels.json")))
key = None
for s in series:
    for d in sorted(p for p in (config.WORK_DIR / s).iterdir() if (p / "district_labels.json").exists()):
        sha = L.page_sha(d)
        key = L.read_key(s, scale=None, tile=config.VISION_TILE, overlap=config.VISION_OVERLAP)
        fresh = L.stored_key(d / "district_labels.json", s) == key
        try:
            out, keys = L._read(d, s, sha, key, None, config.VISION_TILE, config.VISION_OVERLAP, None, write=False)
            print(f"{s}/{d.name}: {len(keys)} answers, {len(out)} labels{'' if fresh else ' (stored key differs)'}",
                  flush=True)
        except llm.LLMCacheMiss as e:
            print(f"{s}/{d.name}: MISS {str(e)[:80]}{'' if fresh else ' (stored key differs)'}", flush=True)
