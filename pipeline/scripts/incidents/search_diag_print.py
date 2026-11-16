"""Print the search diagnosis (search_diagnose.py's diagnose-<q>.json) in a readable form.

    uv run python scripts/incidents/search_diag_print.py [QUARTER ...]"""
import json
import sys

from search_diagnose import OUT

for q in sys.argv[1:] or ["nw", "ne", "sw", "se"]:
    d = json.loads((OUT / f"diagnose-{q}.json").read_text())
    print(f"== {q}: replay equals stage {d['replay_equals_stage']}, energy {d['energy_final']}")
    print("   replay:", "; ".join(f"{lab} {e}" for lab, e in d["replay"]))
    for row in d["incidents"]:
        print(f"-- #{row['id']} {row['lots']} lots {row['table']} -> {row['solved']}: dE {row['dE']} "
              f"({row['answer']}), changed {row['changed']}, incident right after {row['incident_right_after']}")
        print("   transitions:", row["transitions"])
        for e in row["expansion"]:
            print(f"   expansion to {e['code']} ({e['lots']} lots): true {e['true_dE']}, bounded {e['bounded_dE']}; "
                  f"overcharge same-code {e['same_code_pairs']} pairs {e['same_code_over']}, two-code "
                  f"{e['two_code_pairs']} pairs {e['two_code_over']}; cut best {e['cut_best']}; "
                  f"qpbo {e['qpbo_expansion']}")
        for g in row["regions"]:
            print(f"   region {g}")
        print("   whole regions:", row["region_change_is_whole_regions"], "fusion with y:", row["fusion_with_y"])
        for lab, c in row["history_on_change"]:
            print(f"   history on change: {lab}: {c}")
