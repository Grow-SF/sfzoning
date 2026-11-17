"""Spike S2: results.md from the numbers the other steps wrote (results.json, determinism_<R>.json, models/*.json,
the crop and synthetic summaries, v1/results.json).  It opens with the findings: the decision rule computed from the
paired intervals (every label and the unflagged, every seed, against the generative reader's recorded temperatures and
its extrapolated lower ones), the outcome that follows from them, and the caveats (CAVEATS: the choices made after
seeing results, and the review's known limits).  The tables follow.

    uv run python scripts/spikes/s2_ctc/report.py
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RULE_READERS = ("C+dir closed", "B+dir closed", "C closed", "B closed", "A+dir closed")
SEEDS = {"C": ("C", "Cs1", "Cs2"), "B": ("B", "Bs1", "Bs2")}
REASONS = [
    "speed: about 90 crops a second on one CPU thread with the code list's exact scores (a box label, both "
    "directions, about 0.02 s) against the generative reader's median 16 s per box label (its records)",
    "determinism: CPU frames and posteriors byte-identical twice and at 1 vs 8 threads (B and C)",
    "provenance: from-scratch weights (B, C) trained only on Terra's strings and the generative reader's own font; "
    "A's docTR weights (undisclosed training data) are not needed to match the generative reader, though A stays a "
    "little ahead of C at two of C's three seeds",
    "explaining away is direct: the CTC remainder is 'other' and the empty string is 'none' (held-out quarter, C at "
    "T = 1: non-text p(none) 0.985, street names p(other) 0.982, block numbers p(other) 0.999)",
    "the owner accepts a somewhat worse reader than the LLM reads, and C is not worse than the generative reader",
]
CAVEATS = [
    "Chosen after seeing results (review I4): the slanted-height floor in crops.py (found by looking at C's answered "
    "misses in v1); the closed-list framing and the CTC prior weights (+w) (after B's first numbers showed 'other' "
    "taking near-miss reads); the direction rate prior (after D alone flipped 19 of 574 upright answered crops).  The "
    "decision row 'C+dir closed' is the best of four C variants (with/without direction, open/closed).",
    "The first run (v1, sliver crops, same code and measures) was a tie on both measures (the v1 line above).",
    "The generative reader's closed temperature is at its grid's edge (T = 16, -log p still falling).  Its evidence "
    "at lower T can't be computed from the records (the temperature acts inside the sums over placements); the "
    "'extrapolated' rows fit each form's log evidence as a + b / T over the four recorded temperatures and are a "
    "sensitivity check only, not the reader's evidence.  Rescoring it at T = 8 or 11 is the 3.6 h run (not done).",
    "Training variance: seeds 1 and 2 of B and C (same recipe, data and steps) are reported beside seed 0; the "
    "bootstrap intervals hold the labels' sampling only.  Retraining between v1 and v2 alone flipped 18 of 634 "
    "byte-identical crops (review).",
    "The scoring set's own doubt: 102 of 662 answers are flagged by the owner; most of C's lead over the generative "
    "reader is on them.  The bootstrap resamples labels i.i.d. (they cluster by sheet and block) and does not re-pick "
    "the cross-fitted constants inside each resample, so its intervals are somewhat narrow (review M7).",
    "C's held-out-quarter numbers are not fully held out: its synthetic stream's font and prior3 were learned from "
    "the generative reader's 959 training labels, 187 of them in the se quarter (review M1).  The answered-label "
    "comparison is unaffected (none of the 959 is scored or in an answered block).",
    "Known faults in the synthetic stream, left as they are so the seeds replicate C exactly: Terra-box offsets drawn "
    "with the wrong sign (small: means 0.02-0.03 word heights, review M2); line windows cut around ink centres, so a "
    "line often runs through the word's middle and a few crops are solid bands that still carry their string "
    "(review M3), which may feed C's confident misreads of digits crossed by lines.",
    "The closed posteriors differ in their prior: the CTC's has no code prior and sums a code's printed forms "
    "unweighted; the generative reader's applies p(code) and p(form | code) (review M4).",
    "GPU reproducibility was checked on a simplified loop (300 steps at lr / 10, no schedule, clipping or weights), "
    "not the training loop itself (review M5).  The picture sheets cut long crops at the cell's width (review M6).",
    "The by-eye split of the disagreements is the implementer's; the owner's look is the tie-breaker.  C with "
    "'other' kept live (what the solver needs for street names) is a tie with the generative reader at defaults "
    "and was not shown better.",
]


def j(name: str, base: Path | None = None):
    f = (base or C.OUT) / name
    return json.loads(f.read_text()) if f.exists() else None


def ci(x) -> str:
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


def passes(d: dict) -> tuple[bool, bool]:
    """(at least tied on top-1, better on -log p) from a paired difference against the generative reader."""
    return d["top1_ci"][1] >= 0, d["nll_ci"][1] < 0


def rule_rows(pv: dict, names, label: str) -> list[str]:
    out = []
    for name in names:
        if name not in pv:
            continue
        for s in ("all", "unflagged"):
            d = pv[name][s]
            tied, better = passes(d)
            out.append(f"| {name} | {label} | {s} | {d['top1']:+.3f} {ci(d['top1_ci'])} | {'yes' if tied else 'no'} "
                       f"| {d['nll']:+.3f} {ci(d['nll_ci'])} | {'yes' if better else 'no'} |")
    return out


def findings(res: dict) -> list[str]:
    """The findings (see the module): the rule's rows, the seeds, C against A, the outcome, the reasons, the
    caveats."""
    a = res["answered"]
    pv, px = a["paired_vs_generative"], a.get("paired_vs_generative_extrapolated", {})
    out = ["## Findings", "",
           "The decision rule (survey 4.6 step 5) read off the paired intervals against the generative reader, closed "
           "with closed: 'at least tied' when the top-1 difference's interval reaches 0 or above, 'better' when the "
           "-log p difference's interval lies below 0.  'recorded T': the generative reader at its four recorded "
           "temperatures; 'extrapolated T': with lower temperatures extrapolated (approximate, see the caveats).", "",
           "| reader | against the generative reader at | subset | top-1 diff | at least tied | -log p diff | "
           "better |", "|---|---|---|---|---|---|---|"]
    out += rule_rows(pv, RULE_READERS, "recorded T")
    out += rule_rows(px, [n for n in RULE_READERS if n in px][:2], "extrapolated T")
    if "extrapolation_residual_nats" in a:
        e = a["extrapolation_residual_nats"]
        g = a["measures"]["gen-fix1 closed, T extrapolated (approx)"]["all"]
        out += ["", f"The generative reader closed, with extrapolated T (cross-fitted, chose "
                    f"{a['constants']['gen-fix1 closed, T extrapolated (approx)']}): top-1 {g['top1']:.3f} "
                    f"{ci(g['top1_ci'])}, -log p {g['nll']:.3f} {ci(g['nll_ci'])}; by temperature "
                    + "; ".join(f"T={float(t):.3g}: {v['top1']:.3f} / {v['nll']:.3f}"
                                for t, v in e["per_temperature"].items())
                + f".  The a + b / T fit's largest residual per label over the recorded temperatures: median "
                  f"{e['median']} nats, 90% {e['q90']}, max {e['max']}.  So the extrapolated temperatures score "
                  f"worse than T = 16 and the cross-fit keeps 16, but a fit that misses the recorded evidence by "
                  f"nats can't settle it: whether a lower T helps the generative reader stays open (the 'better on "
                  f"-log p' rows are provisional)."]
    # the seeds
    seed_rows, spread = [], {}
    for letter, tags in SEEDS.items():
        tops, nlls = [], []
        for tag in tags:
            name = f"{tag}+dir closed"
            if name not in a["measures"]:
                continue
            m, d = a["measures"][name]["all"], pv.get(name, {})
            u = pv.get(name, {}).get("unflagged")
            tops.append(m["top1"])
            nlls.append(m["nll"])
            seed_rows.append(f"| {name} | {m['top1']:.3f} {ci(m['top1_ci'])} | {m['nll']:.3f} {ci(m['nll_ci'])} | "
                             f"{d['all']['top1']:+.3f} {ci(d['all']['top1_ci'])} | {d['all']['nll']:+.3f} "
                             f"{ci(d['all']['nll_ci'])} | {u['top1']:+.3f} {ci(u['top1_ci'])} | {u['nll']:+.3f} "
                             f"{ci(u['nll_ci'])} |")
        if len(tops) > 1:
            spread[letter] = {"seeds": len(tops), "top1_range": [min(tops), max(tops)],
                              "top1_sd": statistics.stdev(tops), "nll_range": [min(nlls), max(nlls)],
                              "nll_sd": statistics.stdev(nlls)}
    if seed_rows:
        out += ["", "Seeds (training variance; the same recipe, data and steps, seeds 0, 1, 2; direction summed out, "
                    "closed):", "",
                "| reader | top-1 | -log p | top-1 diff vs generative | -log p diff | unflagged top-1 diff | "
                "unflagged -log p diff |", "|---|---|---|---|---|---|---|"] + seed_rows
        gen = a["measures"]["gen-fix1 closed"]["all"]
        for letter, sp in spread.items():
            out.append(f"\n{letter} over {sp['seeds']} seeds: top-1 {sp['top1_range'][0]:.3f}-{sp['top1_range'][1]:.3f}"
                       f" (sd {sp['top1_sd']:.3f}), -log p {sp['nll_range'][0]:.3f}-{sp['nll_range'][1]:.3f} "
                       f"(sd {sp['nll_sd']:.3f}); the generative reader's bootstrap interval for comparison: top-1 "
                       f"{ci(gen['top1_ci'])}, -log p {ci(gen['nll_ci'])}.")
    if a.get("paired_c_vs_a"):
        out += ["", "C against A (docTR's weights):", "", "| pair | subset | top-1 diff | -log p diff |",
                "|---|---|---|---|"]
        for name, d in a["paired_c_vs_a"].items():
            for s in ("all", "unflagged"):
                out.append(f"| {name} | {s} | {d[s]['top1']:+.3f} {ci(d[s]['top1_ci'])} | {d[s]['nll']:+.3f} "
                           f"{ci(d[s]['nll_ci'])} |")
    v1 = j("results.json", C.SPIKE / "v1")
    if v1 and "C+dir closed" in v1["answered"]["paired_vs_generative"]:
        d = v1["answered"]["paired_vs_generative"]["C+dir closed"]
        out += ["", f"v1 (sliver crops, seed 0), C+dir closed against the generative reader: top-1 "
                    f"{d['all']['top1']:+.3f} {ci(d['all']['top1_ci'])}, -log p {d['all']['nll']:+.3f} "
                    f"{ci(d['all']['nll_ci'])}; unflagged {d['unflagged']['top1']:+.3f} "
                    f"{ci(d['unflagged']['top1_ci'])}, {d['unflagged']['nll']:+.3f} {ci(d['unflagged']['nll_ci'])}."]
    # the outcome: the rule passes for a reader only on every label AND the unflagged, at every seed, against the
    # recorded and the extrapolated temperatures
    verdict = {}
    for letter in ("C", "B"):
        checks = []
        for tag in SEEDS[letter]:
            name = f"{tag}+dir closed"
            for table in (pv, px):
                if name in table:
                    checks += [all(passes(table[name][s])) for s in ("all", "unflagged")]
        verdict[letter] = bool(checks) and all(checks)
        verdict[letter + "_checks"] = (sum(checks), len(checks))
    a_pass = "A+dir closed" in pv and all(all(passes(pv["A+dir closed"][s])) for s in ("all", "unflagged"))
    if verdict["C"] or verdict["B"]:
        line = "ADOPT the CTC likelihood: the rule passes on every label and the unflagged, at every seed."
    elif a_pass:
        pairs = {tag: a.get("paired_c_vs_a", {}).get(f"{tag}+dir closed vs A+dir closed") for tag in SEEDS["C"]}
        tied = [tag for tag, ca in pairs.items() if ca is not None and all(
            d[k][0] <= 0 <= d[k][1] for d in (ca["all"], ca["unflagged"]) for k in ("top1_ci", "nll_ci"))]
        below = [tag for tag, ca in pairs.items() if ca is not None and tag not in tied]
        if not tied and not below:
            c_vs_a = "not measured against A"
        elif not below:
            c_vs_a = "tied with A by every interval at every seed"
        else:
            seed = {"C": "seed 0", "Cs1": "seed 1", "Cs2": "seed 2"}
            c_vs_a = (f"tied with A at {', '.join(seed[t] for t in tied) or 'no seed'} and behind A at "
                      f"{', '.join(seed[t] for t in below)} on some interval (table above): A's lead over C is about "
                      f"the size of C's seed spread")
        line = (f"TIE for the from-scratch readers by the owner's rule, to be broken by looking.  Only A (docTR's "
                f"weights, a baseline that can't ship) passes on every label and the unflagged, so by the letter of "
                f"step 5 this is the 'only A wins' case: the remaining gap is pretraining data, and the fix to try is "
                f"the synthetic stream (C has one; its known faults, below, come first).  C is {c_vs_a}.")
    else:
        line = "TIE by the owner's rule, to be broken by looking."
    out += ["", f"**Outcome: {line}**  (C+dir closed passes {verdict['C_checks'][0]} of {verdict['C_checks'][1]} "
                f"checks: every label and the unflagged, each seed, recorded and extrapolated T; B+dir closed "
                f"{verdict['B_checks'][0]} of {verdict['B_checks'][1]}.)",
            "", "Reasons to prefer the CTC reader apart from the rule (reasons, not a rule pass):", ""]
    out += [f"- {r}" for r in REASONS]
    cited = a.get("cited_labels", {})
    if cited:
        names = [n for n in ("gen-fix1 closed", "C+dir closed", "Cs1+dir closed", "Cs2+dir closed")
                 if n in next(iter(cited.values()))]
        out += ["", "The labels the first report cited, top-1 by reader and seed (closed):", "",
                "| label | owner | flagged | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
        for i, row in cited.items():
            out.append(f"| #{i} | {row['owner']} | {'yes' if row['flagged'] else ''} | "
                       + " | ".join(row[n] for n in names) + " |")
        c_names = [n for n in names if n.startswith("C")]
        misses = list(cited.items())[:12]          # score.CITED: the first 12 are C's cited misses
        always = [f"#{i}" for i, row in misses if all(row[n] != row["owner"] for n in c_names)]
        out += ["", f"Of the {len(misses)} labels the first report cited as C's misses (digits crossed by lines), "
                    f"{len(always)} are wrong at all {len(c_names)} of C's seeds ({', '.join(always) or 'none'}); "
                    f"the rest are right at one seed or more, so most of those misses are training variance, not a "
                    f"pattern."]
    out += ["", "Caveats:", ""] + [f"- {c}" for c in CAVEATS] + [""]
    return out


def main():
    res, crops, synth = j("results.json"), j("crops_summary.json"), j("synth_summary.json")
    dets = [d for d in (j(f"determinism_{r}.json") for r in ("B", "C", "A")) if d]
    repro = j("models/repro.json")
    out = ["# Spike S2: a small CTC reader on Terra-box crops", ""]
    out += findings(res) + ["## Data", ""]
    out.append(f"- Maps: {', '.join(crops['maps'])} (the 2000 drawing's three scans). Held out on every map at once, "
               f"through each px's map point: the {crops['hold_quarter']} quarter, the validation area and the "
               f"{crops['answered_blocks']} answered blocks (a training crop touches none of them anywhere in its "
               f"footprint).")
    out.append(f"- Crops {crops['height']} px tall, pads {crops['pad'][0]} / {crops['pad'][1]} word heights along / "
               f"across the line. Terra's miss rate (other passes' boxes with no Terra word, 2000): "
               f"{crops['miss']['rate']:.3f} ({crops['miss']['no_terra_word']}/{crops['miss']['other_pass_boxes']}); "
               f"non-text training crops weighted {1 - crops['miss']['rate']:.3f}.")
    out += ["", "| split / kind | crops |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in sorted(crops["counts"].items()) if k.split("/")[0]]
    if synth:
        out += ["", f"- Synthetic (reader C): {synth['crops']} code-form crops, {synth['codes']} codes; line windows "
                    f"over {synth['overlaid']} (measured share of training code words with run-rule line ink "
                    f"{synth['line_share']['share']:.3f}); speck rate {synth['specks']['rate']:.2e}."]
    out += ["", "## Training", "", "| reader | steps | peak lr | real crops | synthetic | GPU seconds | last loss | "
            "monitor CER (validation area) |", "|---|---|---|---|---|---|---|---|"]
    for r in ("B", "C", "A", "D"):
        m = j(f"models/{r}.json")
        if not m:
            continue
        last = m["steps"][-1]
        out.append(f"| {r} | {m['recipe']['steps']} | {m['recipe']['max_lr']} | {m['real']} | {m['synth']} | "
                   f"{m['train_seconds']:.0f} | {last['loss']:.3f} | {last.get('monitor_cer', '')} |")
    if repro:
        out += ["", f"GPU reproducibility (B's first {repro['steps']} steps, twice, each alone on the GPU): "
                    f"{repro['identical_steps']}/{repro['steps']} losses bit-identical; first difference at step "
                    f"{repro['first_difference']}; largest difference {repro['max_abs_diff']:.3g}."]
    a = res["answered"]
    out += ["", "## The owner's answered labels, box-found", "",
            f"{a['scored']} labels scored ({a['box_found']} box-found; {a['outside_code_list']['n']} answers outside "
            f"the code list: {', '.join(a['outside_code_list']['codes'])}; {a['no_answer']} without an agreed "
            f"answer). Flagged by the owner: {a['flagged']} (the scoring set's own doubt). Generative records: "
            f"fix1 (df61853) {a['generative_records']['fix1']}, fix2 (c626e46, partial) "
            f"{a['generative_records']['fix2']}.", "",
            "| reader | top-1 | top-3 | mean -log p | median -log p | unflagged top-1 | unflagged -log p | "
            "flagged top-1 |", "|---|---|---|---|---|---|---|---|"]
    for name, m in a["measures"].items():
        x, u, f = m["all"], m["unflagged"], m["flagged"]
        out.append(f"| {name} | {x['top1']:.3f} {ci(x['top1_ci'])} | {x['top3']:.3f} {ci(x['top3_ci'])} | "
                   f"{x['nll']:.3f} {ci(x['nll_ci'])} | {x['nll_median']:.3f} | {u['top1']:.3f} {ci(u['top1_ci'])} | "
                   f"{u['nll']:.3f} {ci(u['nll_ci'])} | {f['top1']:.3f} |")
    out += ["", "Saved LLM reads on the same labels (each on the labels it read):", "",
            "| read | labels read | top-1 | unflagged top-1 |", "|---|---|---|---|"]
    for r, m in a["llm_reads"].items():
        if m["all"]["n"]:
            out.append(f"| {r} | {m['all']['n']} | {m['all']['top1']:.3f} {ci(m['all']['top1_ci'])} | "
                       f"{m['unflagged']['top1']:.3f} |")
    out.append("")
    out.append("Yardstick (spec): " + "; ".join(f"{k} {v:.1%}" for k, v in res["llm_yardstick"].items()) + ".")
    out += ["", "Cross-fitted constants (fold: constants chosen on the other fold): " +
            "; ".join(f"{k} {v}" for k, v in a["constants"].items()), ""]
    out += ["### Paired differences from the generative reader's like variant (fix1; closed with closed, cross-fitted "
            "with cross-fitted, defaults with defaults)", "",
            "| reader | against | subset | top-1 diff | -log p diff | it right, gen wrong | it wrong, gen right |",
            "|---|---|---|---|---|---|---|"]
    for name, by in a["paired_vs_generative"].items():
        for s in ("all", "unflagged"):
            d = by[s]
            out.append(f"| {name} | {by['against']} | {s} | {d['top1']:+.3f} {ci(d['top1_ci'])} | {d['nll']:+.3f} "
                       f"{ci(d['nll_ci'])} | {d['a_right_b_wrong']} | {d['a_wrong_b_right']} |")
    out.append("")
    if "fix2_subset" in a:
        f2 = a["fix2_subset"]
        out += [f"### The partial c626e46 run's {f2['n']} labels", "", "| reader | top-1 | mean -log p |",
                "|---|---|---|"]
        out += [f"| {k} | {v['top1']:.3f} {ci(v['top1_ci'])} | {v['nll']:.3f} {ci(v['nll_ci'])} |"
                for k, v in f2.items() if k != "n"]
        out.append("")
    out += ["### Reliability (all scored labels; top-1 probability bins)", ""]
    for name, m in a["measures"].items():
        if not name.endswith(("closed", "cross-fitted +w", "T=1", "default")):
            continue
        out.append(f"- {name}: " + "; ".join(f"{r['bin']} n={r['n']} p={r['mean_p']:.3f} right={r['right']:.3f}"
                                            for r in m["reliability"]))
    q = res["quarter"]
    out += ["", f"## The held-out quarter ({crops['hold_quarter']}, 2000 scan; Terra's strings as truth)", "",
            "Crops: " + ", ".join(f"{k} {v}" for k, v in q["crops"].items()), "",
            "| reader | kind | words | CER | WER |", "|---|---|---|---|---|"]
    for r in res["readers"]:
        for kind, e in q[r]["errors"].items():
            out.append(f"| {r} | {kind} | {e['words']} | {e['cer']:.3f} {ci(e['cer_ci'])} | {e['wer']:.3f} "
                       f"{ci(e['wer_ci'])} |")
    out += ["", "Explaining away (direction summed out, T = 1): mean masses and the top hypothesis by kind", "",
            "| reader | kind | crops | p(code) | p(other) | p(none) | top code | top other | top none |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in res["readers"]:
        for kind, e in q[r]["masses_dir_T1"].items():
            out.append(f"| {r} | {kind} | {e['crops']} | {e['p_code']:.3f} | {e['p_other']:.3f} | {e['p_none']:.3f} "
                       f"| {e['top_is_code']:.3f} | {e['top_is_other']:.3f} | {e['top_is_none']:.3f} |")
    out += ["", "Code top-1 on the held-out quarter against Terra's own string (where it names a code): " +
            "; ".join(f"{r} {q[r]['masses_dir_T1']['zoning code']['top1_vs_terra']:.3f} "
                      f"{ci(q[r]['masses_dir_T1']['zoning code']['top1_vs_terra_ci'])} of "
                      f"{q[r]['masses_dir_T1']['zoning code']['terra_names_code']}" for r in res["readers"])]
    if res.get("direction"):
        d = res["direction"]
        out += ["", f"Direction classifier D on the quarter's {d['quarter_words']} words (Terra's direction as "
                    f"truth): {d['said_upside_down']} called upside down by D alone "
                    f"({d['said_upside_down_by_terra_direction']}), {d['quarter_with_rate_upside_down']} once D's "
                    f"odds sit on the measured half-turn rate of Terra's direction. On the answered labels, by "
                    f"Terra's direction (D alone / with the rate): " + "; ".join(
                        f"{k} {v['D_alone_upside_down']}/{v['with_rate_upside_down']} of {v['n']}"
                        for k, v in d["answered_by_terra_direction"].items()) + "."]
    if dets:
        out += ["", "## Determinism and speed (CPU)", ""]
    for det in dets:
        c = det["crops_per_second_one_thread"]
        out += [f"- Reader {det['reader']}, {det['crops']} answered crops, torch {det['torch']}: one thread twice "
                f"identical: {det['identical_one_thread_twice']}; one vs eight threads identical: "
                f"{det['identical_one_vs_eight_threads']} (largest logit difference "
                f"{det['max_abs_logit_diff_one_vs_eight']:.3g}). Crops per second on one thread: network alone "
                f"{c['network_batch1']} (batch 1), {c['network_batched_by_width']} (batched by width); network plus "
                f"the code list's exact CTC scores {c['network_plus_code_ctc_batch1']}."]
    out += ["", "## Pictures", "",
            "- `disagreements.png`: every answered label where the best from-scratch CTC reader and the generative "
            "reader disagree on top-1, with the owner's code; for judging by eye which is right.",
            "- `reliability.png`: the cross-fitted readers' reliability curves; for seeing over- or under-confidence.",
            "- `look_crops.png`: random training and answered crops; for checking the rectification.",
            "- `synth_examples.png`: synthetic crops of reader C; for checking they look like the scan."]
    (C.OUT / "results.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
