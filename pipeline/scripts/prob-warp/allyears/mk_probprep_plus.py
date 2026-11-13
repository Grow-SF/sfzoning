"""Make ay_probprep_plus.py from ay_probprep.py: the evidence of dm_sides_plus_NN.npz (sides_plus.py), with its weights
and kinds, into S/plus/probin_NN.npz.  usage: mk_probprep_plus.py"""
from pathlib import Path

D = Path(__file__).parent
s = (D / "ay_probprep.py").read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


rep('''"""probprep.py for any series/sheet''', '''"""(sides_plus: the evidence of dm_sides_plus_NN.npz, with its weights and kinds, into S/plus/probin_NN.npz)
probprep.py for any series/sheet''')
rep('''SD = np.load(S + f"dm_sides_{NN}.npz")
pts, ta, tb = side_pixels(T0, SD)''', '''SD = np.load(S + f"dm_sides_plus_{NN}.npz")
pts, ta, tb = T0.to_pixel(SD["pts"]), T0.to_pixel(SD["ta"]), T0.to_pixel(SD["tb"])''')
rep('''np.savez(S + f"probin_{NN}.npz", P=P[ok].astype(np.float32), bin=pbin[ok], sid=SD["sid"][ok], keep=ok, street_px=SW,''',
    '''import os  # noqa: E402
os.makedirs(S + "plus", exist_ok=True)
np.savez(S + f"plus/probin_{NN}.npz", P=P[ok].astype(np.float32), bin=pbin[ok], sid=SD["sid"][ok], keep=ok, street_px=SW,
         wt=SD["wt"][ok].astype(np.float32), kind=SD["kind"][ok],''')
rep('''near = cKDTree(P[ok]).query(gq)[0] <= 100''', '''near = cKDTree(P[ok & (SD["kind"] == 0)]).query(gq)[0] <= 100''')
(D / "ay_probprep_plus.py").write_text(s)
print("ok: ay_probprep_plus.py")
