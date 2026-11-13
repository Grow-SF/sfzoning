"""One year through the whole chain as a task graph (speed E3): each sheet goes correction -> mirror + warp -> chain as
soon as its own inputs exist; the city when every sheet is through; then each sheet's step pictures.  Tasks run
within SLOTS CPU slots (a code fit books NPROC slots); Terra reads (network) run beside them, at most TERRA_P at a
time, and use no slot.  Bigger sheets (more zoning-code boxes, or more block lines before Terra has read them) go
first.  Every task logs to flow/<series>/<task>.log; this prints one line per task start / end and a heartbeat
every 60 s.  Same scripts as the step-by-step run, so the outputs are the same files.

usage: flow.py SERIES   env: SLOTS (10), NPROC (4), TERRA_P (4), CITY_SHEET, SAME2000, NO_RENDER=1"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SER = sys.argv[1]
A = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears")
S = A.parent
PIPE = Path("/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/pipeline")
PY = str(PIPE / ".venv/bin/python")
W = PIPE / "data/work" / SER
SLOTS, NPROC, TERRA_P = int(os.environ.get("SLOTS", 10)), int(os.environ.get("NPROC", 4)), int(os.environ.get("TERRA_P", 4))
LOG = A / "flow" / SER
LOG.mkdir(parents=True, exist_ok=True)
sheets = sorted(d.name for d in W.iterdir() if d.is_dir() and all((d / f).exists() for f in
                ("transform.json", "warped.tif", "blocks_px.geojson", "refine.json")))


def size(sh):
    try:
        return 10_000 + sum(k["kind"] == "zoning code" for k in json.load(open(A / "chain" / SER / sh / "textboxes_gpt-6.1-terra.json")))
    except FileNotFoundError:
        return (W / sh / "blocks_px.geojson").stat().st_size / 1e6


order = sorted(sheets, key=size, reverse=True)
CORR_ENV = {"SAME2000": "1"} if os.environ.get("SAME2000") else {}
PATCH = {"SFZ_CORR": "1", "PYTHONPATH": str(S / "corrpatch"), "SFZMAPS_DATA_DIR": str(S / "data_ay"), "SERIES": SER}
GDAL = "/Users/sbuss/workspace/sfzoning/.claude/worktrees/cache-maybe/.pixi/envs/default/bin"
tasks = {}      # name -> dict(cmd, env, deps, slots, kind, prio)


def add(name, cmd, deps=(), slots=1, kind="cpu", env=None, prio=0):
    tasks[name] = {"cmd": cmd, "deps": set(deps), "slots": slots, "kind": kind, "env": env or {}, "prio": prio}


add("mirror-prepare", [PY, str(A / "build_mirror_ay.py"), SER, "--prepare"], slots=1, prio=1e9)
CORR_ONLY = bool(os.environ.get("CORR_ONLY"))      # maps without printed codes: the correction and warp only
for i, sh in enumerate(order):
    pr = len(order) - i
    if CORR_ONLY:
        add(f"corr:{sh}", ["sh", str(A / "corr.sh"), SER, sh], env=CORR_ENV, prio=pr)
        add(f"mirwarp:{sh}", ["zsh", str(A / "mirwarp1.sh"), SER, sh], deps=[f"corr:{sh}", "mirror-prepare"], prio=pr)
        continue
    tb = A / "chain" / SER / sh / "textboxes_gpt-6.1-terra.json"
    if not tb.exists():
        add(f"terra:{sh}", ["zsh", str(A / "terra1.sh"), SER, sh], kind="net", prio=pr)
    add(f"corr:{sh}", ["sh", str(A / "corr.sh"), SER, sh], env=CORR_ENV, prio=pr)
    add(f"mirwarp:{sh}", ["zsh", str(A / "mirwarp1.sh"), SER, sh], deps=[f"corr:{sh}", "mirror-prepare"], prio=pr)
    # the biggest BIG_N sheets end last: their code fit gets BIG_NPROC workers (E3b)
    npr = int(os.environ.get("BIG_NPROC", NPROC)) if i < int(os.environ.get("BIG_N", 3)) else NPROC
    add(f"chain:{sh}", ["zsh", str(A / "chain1.sh"), SER, sh], deps=[f"mirwarp:{sh}"] + ([f"terra:{sh}"] if not tb.exists() else []),
        slots=npr, env={"NPROC": str(npr)}, prio=pr)
city_env = {"CITY_SHEET": os.environ["CITY_SHEET"]} if os.environ.get("CITY_SHEET") else {}
if not CORR_ONLY:
    add("city", ["zsh", str(A / "cityAY.sh"), SER], deps=[f"chain:{sh}" for sh in sheets], slots=2, env=city_env, prio=1e8)
if not os.environ.get("NO_RENDER") and not CORR_ONLY:
    for i, sh in enumerate(order):
        add(f"render:{sh}", ["zsh", str(A / "renderAY.sh"), SER, sh.rsplit("-p", 1)[1]], deps=["city"], prio=len(order) - i)

done, failed, running = set(), set(), {}     # running: name -> (Popen, t0)
t_start = last_hb = time.time()


def ready(n):
    return n not in done and n not in failed and n not in running and tasks[n]["deps"] <= done


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


say(f"FLOW {SER}: {len(sheets)} sheets, {len(tasks)} tasks, slots {SLOTS}, NPROC {NPROC}, terra {TERRA_P}")
while len(done) + len(failed) < len(tasks):
    for n, (p, t0) in list(running.items()):
        if p.poll() is not None:
            del running[n]
            (done if p.returncode == 0 else failed).add(n)
            say(f"{'END' if p.returncode == 0 else 'FAILED'} {n} {time.time() - t0:.0f}s")
    if any(tasks[n]["deps"] & failed for n in tasks if n not in done | failed | set(running)):
        for n in tasks:
            if n not in done | failed | set(running) and tasks[n]["deps"] & failed:
                failed.add(n)
                say(f"SKIP {n} (an input failed)")
    used = sum(tasks[n]["slots"] for n in running if tasks[n]["kind"] == "cpu")
    nets = sum(1 for n in running if tasks[n]["kind"] == "net")
    for n in sorted((n for n in tasks if ready(n)), key=lambda n: -tasks[n]["prio"]):
        t = tasks[n]
        if t["kind"] == "net":
            if nets >= TERRA_P:
                continue
            nets += 1
        else:
            need = min(t["slots"], SLOTS)
            if used + need > SLOTS and used > 0:
                continue
            used += need
        env = {**os.environ, **t["env"]}
        if n.startswith(("mirwarp", "chain", "city", "render")):
            env = {**env, **PATCH, "PATH": GDAL + ":" + env["PATH"]}
        running[n] = (subprocess.Popen(t["cmd"], cwd=PIPE, env=env, stdout=open(LOG / f"{n.replace(':', '_')}.log", "w"),
                                       stderr=subprocess.STDOUT), time.time())
        say(f"START {n}")
    if time.time() - last_hb > 60:
        last_hb = time.time()
        say(f"HB {len(done)}/{len(tasks)} done, running {sorted(running)}, {time.time() - t_start:.0f}s")
    time.sleep(0.5)
say(f"FLOW {SER} done in {time.time() - t_start:.0f}s; failed: {sorted(failed)}")
sys.exit(1 if failed else 0)
