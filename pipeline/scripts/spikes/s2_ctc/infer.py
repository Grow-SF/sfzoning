"""Spike S2, steps 3-4: every reader's CTC frames on the scoring crops, on the CPU; determinism and speed.

    uv run --with python-doctr --with torch==2.15.0 python scripts/spikes/s2_ctc/infer.py frames A B C
    uv run --with python-doctr --with torch==2.15.0 python scripts/spikes/s2_ctc/infer.py check B

`frames`: for each reader, the logits (float32, frames x classes) of every scoring crop (crops.py's `quarter` and
`answered` splits), as cut (Terra's reading direction) and turned a half turn; and the direction classifier D's
p(upside down) of each crop as cut.  One crop at a time (batch 1: a crop's numbers never depend on its batch), one
thread, torch's deterministic algorithms, float32.  Written to data/solver/experiments/spike-s2/frames_<R>.npz and
direction.npz.

`check`: the determinism and speed check (survey section 4.6 step 4) on the answered crops: the frames twice on one
thread and once on eight, compared byte for byte (and the code posteriors' JSON); crops per second on one thread, the
network alone and with the code list's exact CTC scores, batch 1 and batched by width.  determinism_<R>.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import train as T  # noqa: E402

SPLITS = ("quarter", "answered")


def load(reader: str):
    model = T.build(reader) if reader != "A" else _bare_A()
    model.load_state_dict(torch.load(C.OUT / "models" / f"{reader}.pt", map_location="cpu"))
    model.eval()
    return model


def _bare_A():
    """A's architecture without downloading anything (its weights come from A.pt)."""
    from doctr.models import crnn_mobilenet_v3_small
    return crnn_mobilenet_v3_small(pretrained=False, pretrained_backbone=False, vocab=C.VOCAB)


def settle(threads: int) -> None:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def frames_of(model, crops: list[np.ndarray]) -> list[np.ndarray]:
    """The logits of each crop, one at a time."""
    out = []
    with torch.inference_mode():
        for c in crops:
            x, n = C.to_input([c])
            out.append(C.forward(model, x)[0, :n[0]].numpy().copy())
    return out


def flips_of(model, crops: list[np.ndarray]) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for c in crops:
            x, _ = C.to_input([c])
            out.append(float(torch.sigmoid(model(x))[0]))
    return np.array(out)


def pack(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lens = np.array([a.shape[0] for a in arrays], np.int64)
    return np.concatenate(arrays, 0), np.concatenate([[0], np.cumsum(lens)])


def scoring_crops():
    crops = C.load_crops()
    m = crops.meta
    idx = np.flatnonzero(m["split"].isin(SPLITS).to_numpy())
    return crops, idx


def frames(readers: list[str]) -> None:
    settle(1)
    crops, idx = scoring_crops()
    cut = [crops[i] for i in idx]
    turned = [np.ascontiguousarray(np.rot90(c, 2)) for c in cut]
    for r in readers:
        t0 = time.perf_counter()
        if r == "D":
            p = flips_of(load("D"), cut)
            np.savez(C.OUT / "direction.npz", index=idx, p_flip=p)
        else:
            model = load(r)
            f0, o0 = pack(frames_of(model, cut))
            f1, o1 = pack(frames_of(model, turned))
            np.savez(C.OUT / f"frames_{r}.npz", index=idx, cut=f0, cut_off=o0, turned=f1, turned_off=o1)
        print(f"{r}: {len(idx)} crops x {1 if r == 'D' else 2} in {time.perf_counter() - t0:.0f} s", flush=True)


def _digest(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.tobytes())
    return h.hexdigest()


def check(reader: str) -> None:
    crops, idx = scoring_crops()
    m = crops.meta
    ans = idx[(m["split"].to_numpy()[idx] == "answered")]
    cut = [crops[i] for i in ans]
    forms = json.loads((C.INPUTS / "prior3.json").read_text())["forms"]
    model = load(reader)
    runs = {}
    for name, threads in (("one_thread_a", 1), ("one_thread_b", 1), ("eight_threads", 8)):
        settle(threads)
        t0 = time.perf_counter()
        f = frames_of(model, cut)
        t_net = time.perf_counter() - t0
        t0 = time.perf_counter()
        posts = [C.code_posterior(F.log_softmax(torch.from_numpy(a).double(), -1), forms) for a in f]
        t_ctc = time.perf_counter() - t0
        runs[name] = {"threads": threads, "frames_sha256": _digest(f),
                      "posteriors_sha256": hashlib.sha256(json.dumps(posts, sort_keys=True).encode()).hexdigest(),
                      "net_seconds": round(t_net, 2), "ctc_seconds": round(t_ctc, 2), "frames": f}
        print(name, {k: v for k, v in runs[name].items() if k != "frames"}, flush=True)
    a, b, e = runs["one_thread_a"], runs["one_thread_b"], runs["eight_threads"]
    max_diff = max(float(np.abs(x - y).max()) for x, y in zip(a["frames"], e["frames"]))
    # batched by width (crops of one padded width together), one thread: the speed only
    settle(1)
    order = sorted(range(len(cut)), key=lambda i: C.padded(cut[i]).shape[1])
    t0 = time.perf_counter()
    with torch.inference_mode():
        k = 0
        while k < len(order):
            w = C.padded(cut[order[k]]).shape[1]
            j = k
            while j < len(order) and C.padded(cut[order[j]]).shape[1] == w and j - k < 64:
                j += 1
            x, _ = C.to_input([cut[i] for i in order[k:j]])
            C.forward(model, x)
            k = j
    t_batched = time.perf_counter() - t0
    n = len(cut)
    res = {"reader": reader, "crops": n, "torch": torch.__version__,
           "identical_one_thread_twice": a["frames_sha256"] == b["frames_sha256"]
           and a["posteriors_sha256"] == b["posteriors_sha256"],
           "identical_one_vs_eight_threads": a["frames_sha256"] == e["frames_sha256"]
           and a["posteriors_sha256"] == e["posteriors_sha256"],
           "max_abs_logit_diff_one_vs_eight": max_diff,
           "runs": {k: {kk: vv for kk, vv in v.items() if kk != "frames"} for k, v in runs.items()},
           "crops_per_second_one_thread": {"network_batch1": round(n / a["net_seconds"], 1),
                                           "network_plus_code_ctc_batch1": round(n / (a["net_seconds"]
                                                                                    + a["ctc_seconds"]), 1),
                                           "network_batched_by_width": round(n / t_batched, 1)},
           "note": "batch 1 is what the frames step uses (a crop's numbers never depend on its batch)"}
    (C.OUT / f"determinism_{reader}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    what, readers = sys.argv[1], sys.argv[2:]
    if what == "frames":
        frames(readers)
    elif what == "check":
        check(readers[0])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
