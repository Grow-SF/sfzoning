"""Final code per printed label: a person's answer, else Terra and Sol agreeing, else the old read
(flagged for a person).  Writes codes_final.json and reorders the code-check page so the
unchecked disagreements come first."""
import glob
import json
import re
from collections import Counter

SP = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/"


def norm(s):
    s = re.sub(r"\s+", "", (s or "").upper())
    return "NONE" if s in ("", "NONE") else s


cases = json.load(open(SP + "codecheck/cases.json"))
T = {r["id"]: r for r in json.load(open(SP + "reread_gpt-6.1-terra_high_all.json"))}
S = {r["id"]: r for r in json.load(open(SP + "reread_gpt-6.1-sol_high_all.json"))}
H = {f.split("/")[-1][:-5]: json.load(open(f)) for f in glob.glob(SP + "truth2/reads/*.json")}
# the leader-check page's typed codes and "not here" answers, by printed place
lc = {c["id"]: c for c in json.load(open(SP + "labeler/cases.json"))}
L1 = {}
for f in glob.glob(SP + "truth/labels/*.json"):
    a, c = json.load(open(f)), lc[f.split("/")[-1][:-5]]
    if a["answer"] == "nolabel" or a.get("seen"):
        L1[(c["sheet"], round(c["x"]), round(c["y"]))] = a

out, src = [], Counter()
for c in cases:
    t, s = norm(T[c["id"]]["reread"]), norm(S[c["id"]]["reread"])
    h = H.get(c["id"])
    l1 = L1.get((c["sid"], round(c["x"]), round(c["y"])))
    if h and h["answer"] in ("ok", "fix", "nolabel"):
        code = {"ok": c["code"], "fix": h.get("code"), "nolabel": None}[h["answer"]]
        how = "person"
    elif l1:
        code = None if l1["answer"] == "nolabel" else l1["seen"]
        how = "person"
    elif t == s:
        code = None if t == "NONE" else T[c["id"]]["reread"]
        how = "agree"
    else:
        code, how = c["code"], "open"
    src[how] += 1
    out.append({**c, "final": code, "how": how, "terra": t, "sol": s,
                "changed": norm(code) != norm(c["code"])})
json.dump(out, open(SP + "codes_final.json", "w"), indent=1)
print(src, "| changed from the old read:", sum(o["changed"] for o in out),
      "| dropped as not a code:", sum(o["final"] is None for o in out))
print("agree changes:", Counter((o["code"], o["final"]) for o in out if o["how"] == "agree" and o["changed"]).most_common(15))

# page order: unchecked disagreements first, then everything else as before
open_ids = [o for o in cases if next(x for x in out if x["id"] == o["id"])["how"] == "open"]
rest = [o for o in cases if o not in open_ids]
for o in cases:
    o["flag"] = o in open_ids
json.dump(open_ids + rest, open(SP + "codecheck/cases.json", "w"), separators=(",", ":"))
print("page: first", len(open_ids), "are the open ones")
