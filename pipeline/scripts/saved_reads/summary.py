"""Per read in a package: answers, how matched, models and efforts, when asked, images, sheets, templates, scripts.
usage: python -m saved_reads.summary [PACKAGE]   (default pipeline/saved-reads)"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

pkg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "saved-reads"
for f in sorted((pkg / "reads").glob("*.jsonl.gz")):
    rs = [json.loads(line) for line in gzip.open(f, "rt")]
    asked = sorted(r["asked"] for r in rs)
    print(f"## {f.name.removesuffix('.jsonl.gz')}: {len(rs)} answers; {dict(Counter(r['matched'] for r in rs))}")
    print(f"   models {dict(Counter((r['model'], r.get('effort')) for r in rs))}")
    print(f"   asked {asked[0]} .. {asked[-1]}")
    print(f"   images {len({r['image'] for r in rs if r.get('image')})}; series {sorted({str(r.get('series')) for r in rs})}; "
          f"sheets {len({r.get('sheet') for r in rs if r.get('sheet')})}")
    print(f"   templates {dict(Counter(r.get('prompt') for r in rs))}")
    print(f"   scripts {sorted({r['script']['source'] for r in rs})}")
