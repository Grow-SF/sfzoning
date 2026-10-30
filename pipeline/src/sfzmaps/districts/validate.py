"""Validation: agreement between two district readings (2000-digital-zoning vs DataSF today)
and accuracy against hand-read answer keys (tests/fixtures/district_keys/<series>.csv).

Agreement is computed over the parcels both frames contain (by blklot):
  category_agreement      same story category, over parcels coded (category != unknown) in both;
  category_agreement_all  the same over every common parcel (an unknown counts as a miss);
  code_agreement          same normalised code, over parcels coded in both;
  family_agreement        same code family (categories.family_for), over parcels coded in both;
  by_family / by_category the per-family (per-category) share of a's parcels that agree;
  family_confusion        a family -> {b family: count} for disagreeing parcels;
  category_confusion      a category -> {b category: count} for disagreeing parcels.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sfzmaps.districts.categories import family_for
from sfzmaps.districts.rules import normalise_code


def _norm(code) -> str | None:
    if not isinstance(code, str) or not code.strip():
        return None
    return normalise_code(code, 0) or code.strip().upper()


def _share(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def _confusion(x: pd.Series, y: pd.Series) -> dict:
    out: dict = {}
    for (i, j), n in pd.crosstab(x, y).stack().items():
        if i != j and n:
            out.setdefault(str(i), {})[str(j)] = int(n)
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in sorted(out.items())}


def _by(group: pd.Series, ok: pd.Series) -> dict:
    out = {}
    for k, g in ok.groupby(group):
        out[str(k)] = {"n": int(len(g)), "agree": int(g.sum()), "share": _share(int(g.sum()), len(g))}
    return out


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Category / code / family agreement of a vs b (columns blklot, code, category)."""
    m = a[["blklot", "code", "category"]].drop_duplicates("blklot").merge(
        b[["blklot", "code", "category"]].drop_duplicates("blklot"), on="blklot", suffixes=("_a", "_b"))
    n_common = len(m)
    known = m[(m["category_a"] != "unknown") & (m["category_b"] != "unknown")].copy()
    n = len(known)
    cat_ok = known["category_a"] == known["category_b"]
    code_ok = known["code_a"].map(_norm) == known["code_b"].map(_norm)
    known["fam_a"] = known["code_a"].map(family_for)
    known["fam_b"] = known["code_b"].map(family_for)
    fam_ok = known["fam_a"] == known["fam_b"]
    return {
        "n_a": int(a["blklot"].nunique()), "n_b": int(b["blklot"].nunique()),
        "n_common": n_common, "n_compared": n,
        "category_agreement": _share(int(cat_ok.sum()), n),
        "category_agreement_all": _share(int((m["category_a"] == m["category_b"]).sum()
                                             - ((m["category_a"] == "unknown")
                                                & (m["category_b"] == "unknown")).sum()), n_common),
        "code_agreement": _share(int(code_ok.sum()), n),
        "family_agreement": _share(int(fam_ok.sum()), n),
        "by_category": _by(known["category_a"], cat_ok),
        "by_family": _by(known["fam_a"], fam_ok),
        "category_confusion": _confusion(known["category_a"], known["category_b"]) if n else {},
        "family_confusion": _confusion(known["fam_a"], known["fam_b"]) if n else {},
    }


def key_report(df: pd.DataFrame, key_csv: Path | str) -> pd.DataFrame:
    """One row per answer-key parcel with a code: blklot, key, read, ok (normalised codes)."""
    key = pd.read_csv(key_csv, dtype=str, keep_default_na=False)
    key = key[key["code"].str.strip() != ""]
    read = df.drop_duplicates("blklot").set_index("blklot")["code"]
    got = key["blklot"].map(read)
    ok = [_norm(k) is not None and _norm(k) == _norm(g) for k, g in zip(key["code"], got)]
    return pd.DataFrame({"blklot": key["blklot"].to_numpy(), "key": key["code"].to_numpy(),
                         "read": got.to_numpy(), "ok": ok})


def key_accuracy(df: pd.DataFrame, key_csv: Path | str) -> float:
    """Share of the answer key's parcels whose read code matches (a missing parcel is wrong)."""
    r = key_report(df, key_csv)
    return _share(int(r["ok"].sum()), len(r))
