"""Spike S2: results.md from the numbers the other steps wrote (results.json, determinism.json, models/*.json, the crop
and synthetic summaries).  The findings' prose sits in FINDINGS (written after reading the numbers); the tables are
made from the files.

    uv run python scripts/spikes/s2_ctc/report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FINDINGS = C.OUT / "findings.md"      # hand-written findings, put at the top when present


def j(name: str):
    f = C.OUT / name
    return json.loads(f.read_text()) if f.exists() else None


def ci(x) -> str:
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


RULE_READERS = ("B+dir closed", "C+dir closed", "B closed", "C closed", "A+dir closed")


def decision(res) -> list[str]:
    """The decision rule (survey 4.6 step 5), read off the paired intervals against the generative reader (closed with
    closed: the like-for-like measure, see score.py): "at least tied" on top-1 when the difference's interval reaches
    0 or above, "better" on -log p when its interval lies below 0; on every label and on the unflagged (the scoring
    set's own doubt)."""
    pv = res["answered"]["paired_vs_generative"]
    out = ["## Decision rule (survey 4.6 step 5), read off the intervals", "",
           "| reader | subset | top-1 diff | at least tied | -log p diff | better |", "|---|---|---|---|---|---|"]
    for name in RULE_READERS:
        if name not in pv:
            continue
        for s in ("all", "unflagged"):
            d = pv[name][s]
            tied = d["top1_ci"][1] >= 0
            better = d["nll_ci"][1] < 0
            out.append(f"| {name} (vs {pv[name]['against']}) | {s} | {d['top1']:+.3f} {ci(d['top1_ci'])} | "
                       f"{'yes' if tied else 'no'} | {d['nll']:+.3f} {ci(d['nll_ci'])} | {'yes' if better else 'no'} |")
    out += ["", "A is a baseline only (docTR's weights' training data is undisclosed).", ""]
    return out


def main():
    res, crops, synth = j("results.json"), j("crops_summary.json"), j("synth_summary.json")
    dets = [d for d in (j(f"determinism_{r}.json") for r in ("B", "C", "A")) if d]
    repro = j("models/repro.json")
    out = ["# Spike S2: a small CTC reader on Terra-box crops", ""]
    if FINDINGS.exists():
        out += [FINDINGS.read_text().strip(), ""]
    out += decision(res) + ["## Data", ""]
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
