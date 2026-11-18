"""One table of every series' solve against its published map, from each series' score.json ($0, reads only).

    uv run python scripts/all_years/summary.py

Writes data/solver/experiments/all-years/summary.md and summary.json."""
import json

from sfzmaps.solver import paths

ROOT = paths.solver_dir("x").parent


def main():
    rows = []
    for f in sorted(ROOT.glob("*/score.json")):
        s = json.loads(f.read_text())
        t = s.get("table_solve")
        p = s.get("table_published_site")
        root = f.parent
        ev = json.loads((root / "walls" / "evaluation.json").read_text()) if (root / "walls" / "evaluation.json").is_file() else {}
        tr = json.loads((root / "labels" / "trust.json").read_text()) if (root / "labels" / "trust.json").is_file() else {}
        rows.append({
            "series": s["series"], "prints": tr.get("prints"), "reads": tr.get("read_set"),
            "width_scale": ev.get("width_scale"),
            "agree_lots": s["agree_share_lots"], "agree_area": s["agree_share_area"],
            "differing_lots": s["differing_lots"], "differing_areas": s["differing_areas"],
            "own_labels_solve/published": (lambda o: o and f"{o.get('solve')}/{o.get('published')} of {o.get('labels')}")(
                s.get("own_labels_right")),
            "overrides": f"{s['overrides'].get('agree', '-')}/{s['overrides'].get('on_solve_lots', 0)}",
            "label_fixes": f"{s['label_fixes'].get('agree', '-')}/{s['label_fixes'].get('rows', 0)}",
            "label_fixes_published": f"{s['label_fixes_published'].get('agree', '-')}/"
                                     f"{s['label_fixes_published'].get('rows', 0)}",
            "table_solve": t and f"{t['wrong_lots']:,} wrong / {t['labels_right']:.1%} / {t['incidents']:,} inc",
            "table_published": p and f"{p['wrong_lots']:,} wrong / {p['labels_right']:.1%} / {p['incidents']:,} inc",
            "largest": "; ".join(f"{a['published']}->{a['solve']} {a['lots']} lots" for a in s["largest_areas"][:3]),
        })
    cols = list(rows[0]) if rows else []
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    (ROOT / "summary.md").write_text("\n".join(md) + "\n")
    (ROOT / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
