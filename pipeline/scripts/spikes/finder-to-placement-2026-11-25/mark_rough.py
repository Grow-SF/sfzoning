# One-off check, not pipeline code: records the copied rough.json as fresh so `sfzmaps run` takes it as current (step 2 of the rough-check).
"""Step 2 (SFZMAPS_DATA_DIR=<new root>). Record rough.json's freshness exactly as runner.run_sheet's stage("rough", rj,
[lj, bj, *_ref_inputs()], ...) does after do_rough, so `sfzmaps run` takes the copied rough as current."""
import os
from pathlib import Path
from sfzmaps import config, fresh, runner
root = Path(os.environ["SFZMAPS_DATA_DIR"]).resolve()
assert config.DATA_DIR.resolve() == root and root.name == "finder" and root.parent.name == "rough-check", root
d = config.WORK_DIR / "1921" / "20180102094348509"
rj, lj, bj = d / "rough.json", d / "labels.json", d / "blocks_px.geojson"
ins = [lj, bj, *runner._ref_inputs()]
print("rough inputs:", [str(p) for p in ins])
print("fresh before record:", fresh.is_fresh(rj, ins))
fresh.record(rj, ins)
print("fresh after record:", fresh.is_fresh(rj, ins))
print(fresh.sidecar(rj).read_text())
