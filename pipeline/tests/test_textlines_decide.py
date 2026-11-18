"""scripts/textlines/decide.py: the verdict between a control and a variant, by the rule written before the runs, on
fake run folders (synthetic blocks.npz in the gate's layout)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "textlines"))
import decide  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sfzmaps.solver.textnet import linegate  # noqa: E402

BLOCKS = 300
PX = 1000.0
NE = 1                                   # paths.QUARTERS.index("ne")
# The control's three seeds, near Task 5's NE values; the spread between seeds (0.01) is the seed noise.
CONTROL = [{"m1a_in_box": 0.40, "m1b_in_box": 0.52, "m1c": 0.050, "m2": 0.965, "m2_clean_words": 0.930},
           {"m1a_in_box": 0.41, "m1b_in_box": 0.53, "m1c": 0.055, "m2": 0.970, "m2_clean_words": 0.935},
           {"m1a_in_box": 0.405, "m1b_in_box": 0.525, "m1c": 0.052, "m2": 0.960, "m2_clean_words": 0.925}]
OLD_NE = {"m1a_in_box": 0.63, "m1b_in_box": 0.82, "m1c": 0.025, "m2": 0.968, "m2_clean_words": 0.957}


def fake_run(folder: Path, rates: dict, seed: int, quarter: str = "ne") -> Path:
    """A folder as `linegate.write` leaves it: every measure's sums per block, the share called near `rates[m]` (0.5
    for a measure not given), each block off by an amount shared by every run (the block's own difficulty) plus a
    little of the run's own."""
    shared = np.random.default_rng(99)
    own = np.random.default_rng(seed)
    arrays = {}
    for m in linegate.MEASURES:
        r = np.clip(rates.get(m, 0.5) + shared.normal(0, 0.01, BLOCKS) + own.normal(0, 0.01, BLOCKS), 0, 1)
        called = np.round(r * PX)
        arrays |= {f"{m}.quarter": np.full(BLOCKS, NE, np.int8), f"{m}.block": np.arange(BLOCKS, dtype=np.int32),
                   f"{m}.px": np.full(BLOCKS, PX), f"{m}.called_05": called, f"{m}.called_09": called * 0.9,
                   f"{m}.chance": called, f"{m}.log_loss": PX - called}
    folder.mkdir(parents=True)
    np.savez(folder / linegate.BLOCKS_FILE, block_key=np.array("test blocks"), **arrays)
    (folder / "run.json").write_text(json.dumps({"quarter": quarter, "seed": seed}))
    return folder


def arm(root: Path, name: str, seeds_rates: list[dict]) -> list[Path]:
    return [fake_run(root / f"{name}-ne-s{s}", r, 10 * s + len(name)) for s, r in enumerate(seeds_rates)]


def changed(rates: list[dict], **new) -> list[dict]:
    return [r | new for r in rates]


def flat_lines(rates: list[dict]) -> list[dict]:
    """The control's line measures a little higher (0.003, far under the noise), so they cannot fall by chance."""
    return [r | {"m1a_in_box": r["m1a_in_box"] + 0.003, "m1b_in_box": r["m1b_in_box"] + 0.003} for r in rates]


ADOPTABLE = changed(CONTROL, m1a_in_box=0.30, m1b_in_box=0.45, m1c=0.026, m2=0.968, m2_clean_words=0.955)


@pytest.fixture
def control(tmp_path):
    return arm(tmp_path / "runs", "control", CONTROL)


@pytest.fixture
def baselines(tmp_path):
    return {"old-ne": fake_run(tmp_path / "baselines" / "old-ne", OLD_NE, 1000)}


def test_better_lines_with_the_same_text_and_the_eye_ok_is_the_variant(tmp_path, control, baselines):
    variant = arm(tmp_path / "runs", "composites", changed(CONTROL, m1a_in_box=0.30, m1b_in_box=0.45))
    v = decide.verdict(control, variant, baselines, True, boot=200)
    assert v["verdict"] == "variant"
    assert all(v["conditions"].values())
    d = v["variant_vs_control"]["m1a_in_box"]
    assert d["diff"] == pytest.approx(-0.105, abs=0.005) and d["hi"] < 0
    # the noise is the control's seed spread (0.01), wider here than the paired interval
    assert v["variant_vs_control"]["m2"]["noise"] == pytest.approx(0.010, abs=2e-3)
    assert v["variant_vs_control"]["m2"]["seed_noise"] >= v["variant_vs_control"]["m2"]["half_width"]


def test_less_text_beyond_the_noise_is_mixed(tmp_path, control, baselines):
    variant = arm(tmp_path / "runs", "composites", changed(CONTROL, m1a_in_box=0.30, m2=0.90))
    v = decide.verdict(control, variant, baselines, True, boot=200)
    assert v["verdict"] == "mixed"
    assert v["conditions"]["m1a_in_box falls"] and not v["conditions"]["m2 held"]


def test_clean_words_lost_beyond_the_noise_is_mixed(tmp_path, control, baselines):
    """The controller's ruling: m2_clean_words guards the verdict as m2 does."""
    variant = arm(tmp_path / "runs", "composites", changed(CONTROL, m1a_in_box=0.30, m2_clean_words=0.88))
    v = decide.verdict(control, variant, baselines, True, boot=200)
    assert v["verdict"] == "mixed"
    assert v["conditions"]["m2 held"] and not v["conditions"]["m2_clean_words held"]


def test_no_eye_yet_is_pending_and_says_what_the_numbers_say(tmp_path, control, baselines):
    variant = arm(tmp_path / "runs", "composites", changed(CONTROL, m1a_in_box=0.30, m1b_in_box=0.45))
    v = decide.verdict(control, variant, baselines, None, boot=200)
    assert v["verdict"] == "pending-eye"
    assert v["numbers_say"] == "variant"
    no = decide.verdict(control, variant, baselines, False, boot=200)
    assert no["verdict"] == "mixed"          # the numbers gain, the eye says no: a split


def test_m1b_down_with_m1a_flat_is_mixed(tmp_path, control, baselines):
    variant = arm(tmp_path / "runs", "composites", changed(flat_lines(CONTROL), m1b_in_box=0.40))
    v = decide.verdict(control, variant, baselines, True, boot=200)
    assert v["verdict"] == "mixed"
    assert v["variant_vs_control"]["m1b_in_box"]["hi"] < 0 and not v["conditions"]["m1a_in_box falls"]


def test_a_variant_that_gains_nothing_leaves_the_control(tmp_path, control, baselines):
    variant = arm(tmp_path / "runs", "composites", changed(flat_lines(CONTROL), m1c=0.10))
    v = decide.verdict(control, variant, baselines, True, boot=200)
    assert v["verdict"] == "control"


def test_adoptable_needs_every_condition_against_the_baseline(tmp_path, control, baselines):
    """The control loses m1c to old-ne (0.05 vs 0.025) beyond the noise; a variant that keeps m1c, m2 and the clean
    words at old-ne's level and lowers both line measures is adoptable, pending until the audit is counted by eye."""
    variant = arm(tmp_path / "runs", "composites", ADOPTABLE)
    v = decide.verdict(control, variant, baselines, True, boot=200)
    c, w = v["adoptable"]["control"], v["adoptable"]["variant"]
    assert c["old-ne"]["conditions"]["m1c held"] is False and c["adoptable"] is False
    assert w["old-ne"]["conditions"]["m1a_in_box falls"] and w["old-ne"]["conditions"]["m1b_in_box falls"]
    assert w["old-ne"]["conditions"]["audit held"] is None and w["adoptable"] is None


def write_audit(f: Path, missed: int, pieces: int = 1000, drawn: int = 64) -> Path:
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"letter_sized_pieces": pieces, "missed_below_0.5": missed,
                             "items": [{"sheet": "s", "x": i, "y": 0} for i in range(drawn)]}))
    return f


def test_the_audit_holds_unless_more_missed_text_than_the_baselines_upper_bound(tmp_path, control, baselines):
    variant = arm(tmp_path / "runs", "composites", ADOPTABLE)
    base = write_audit(decide.baseline_audit_file(baselines["old-ne"], "ne"), missed=900)
    runs = [write_audit(r / "audit.json", missed=800) for r in variant]
    # 2 of the baseline's 64 drawn are text: about 28 missed text pieces, at most 0.108 x 900 = 97 at 95%
    few = {base: 2} | {r: 2 for r in runs}
    v = decide.verdict(control, variant, baselines, True, audit_counts=few, boot=200)
    a = v["adoptable"]["variant"]["old-ne"]
    assert a["audit"]["arm"]["missed_text"] == pytest.approx(800 * 6 / 192)
    assert a["audit"]["baseline"]["missed_text_ci"][1] == pytest.approx(900 * 0.1084, rel=0.01)
    assert a["conditions"]["audit held"] is True and v["adoptable"]["variant"]["adoptable"] is True
    many = {base: 2} | {r: 20 for r in runs}  # 250 missed text pieces
    v = decide.verdict(control, variant, baselines, True, audit_counts=many, boot=200)
    assert v["adoptable"]["variant"]["old-ne"]["conditions"]["audit held"] is False
    assert v["adoptable"]["variant"]["adoptable"] is False


def test_a_single_seed_comparison_takes_its_noise_from_the_noise_runs(tmp_path, control):
    """On SW one seed each: the seed noise comes from the control's NE seeds."""
    sw_c = [fake_run(tmp_path / "runs" / "control-sw-s0", CONTROL[0], 1, "sw")]
    sw_v = [fake_run(tmp_path / "runs" / "composites-sw-s0", CONTROL[0] | {"m1a_in_box": 0.30}, 2, "sw")]
    v = decide.verdict(sw_c, sw_v, {}, True, noise_runs=control, boot=200)
    assert v["variant_vs_control"]["m2"]["seed_noise"] == pytest.approx(0.010, abs=2e-3)
    assert v["adoptable"]["variant"]["adoptable"] is None          # no baseline given: nothing to adopt against
    alone = decide.verdict(sw_c, sw_v, {}, True, boot=200)
    assert alone["variant_vs_control"]["m2"]["seed_noise"] == 0.0
