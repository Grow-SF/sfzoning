"""probfit.py takes added evidence (sides_plus.py) when its probin has it: each point's data term weighted by its
`wt` (block sides 1); the outlier / spread estimate (EM) and the reported measure (share within 1/2/4 px) on the block
sides alone (kind 0), as before, so runs stay comparable.  Without wt/kind in probin: unchanged.
usage: mk_probfit_plus.py  (edits the scratchpad's probfit.py in place; a copy is kept as probfit_before_plus.py)"""
import shutil
from pathlib import Path

P = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/probfit.py")
shutil.copy(P, P.with_name("probfit_before_plus.py"))
s = P.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:70])
    s = s.replace(old, new)


rep('''N = len(Pp)
''', '''N = len(Pp)
# added evidence (sides_plus.py): each point's weight, and which points are block sides (kind 0)
if "wt" in PI.files:
    WPT = WPT * torch.from_numpy(PI["wt"].astype(np.float32))
    NOBS = float(WPT.sum()) + NG
BASE = (PI["kind"] == 0) if "kind" in PI.files else np.ones(N, bool)
''')
rep('''def nearest_d(xy):
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return T8[PB, yi, xi].astype(np.float64)''', '''def nearest_d(xy, m=None):
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return T8[PB if m is None else PB[m], yi, xi].astype(np.float64)''')
rep('''def share(xy):
    d = nearest_d(xy)''', '''def share(xy):
    d = nearest_d(xy[BASE], BASE)''')
rep('''eps, sig = em(nearest_d(Pp), 0.5, SW / 4)''', '''eps, sig = em(nearest_d(Pp[BASE], BASE), 0.5, SW / 4)''')
rep('''    eps, sig = em(nearest_d(Pp + u), eps, sig)''', '''    eps, sig = em(nearest_d((Pp + u)[BASE], BASE), eps, sig)''')
P.write_text(s)
import ast  # noqa: E402
ast.parse(s)
print("probfit.py: weights and block-side-only estimates in; parses")
