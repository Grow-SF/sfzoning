"""`sfzmaps release`: everything that goes public after a merge, in one command.

Checks (tree clean, HEAD is origin/main, DVC data matches dvc.lock, this commit's cache is
shipped), then the GIS downloads export, the tile publish and the downloads publish (site/scripts,
with the Blob token from the env file `ship` uses), then a branch and pull request carrying the
two manifests they changed (site/lib/published.json, site/lib/downloads.json). Merging that PR
deploys. All git, gh, node and DVC calls go through `Deps`, so tests inject them.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from sfzmaps import config, ship

MANIFESTS = ["site/lib/published.json", "site/lib/downloads.json"]
SCRIPTS = ["publish-tiles.mjs", "publish-downloads.mjs"]
PR_FOOTER = "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
DVC_STATUS = ["uv", "tool", "run", "--from", "dvc==3.67.1", "dvc", "status", "--json"]


class ReleaseError(Exception):
    pass


def _run(cmd: list[str], cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _downloads_run(log: Callable[[str], None]) -> dict:
    from sfzmaps import downloads
    return downloads.run(log=log)


def _tools_outside_pixi() -> list[str]:
    """The system tools (GDAL and the rest) not from the repository's pixi environment."""
    from sfzmaps import toolchain
    return [f"{t.name} ({t.path or 'missing'})" for t in toolchain.not_from_env()]


def _today() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


@dataclass
class Deps:
    git: Callable[..., str] = ship._git                              # git(*args) -> stdout
    node: Callable[..., subprocess.CompletedProcess] = ship._node    # node(script, args); reads the env file
    run: Callable[..., subprocess.CompletedProcess] = _run           # run(cmd, cwd=None)
    check_shipped: Callable[[str], list[str]] = ship.check_shipped   # the `Cache shipped` workflow's check
    downloads: Callable[[Callable[[str], None]], dict] = _downloads_run  # downloads(log)
    toolchain: Callable[[], list[str]] = _tools_outside_pixi           # tools outside pixi
    env_file: Callable[[], object] = ship.env_file_path              # raises ShipError when missing
    today: Callable[[], str] = _today
    echo: Callable[[str], None] = print


@dataclass
class Summary:
    files: int = 0
    bytes: int = 0
    superseded: int = 0
    deleted: int = 0
    pr_url: str | None = None
    branch: str | None = None


def _why(e: Exception) -> str:
    return (getattr(e, "stderr", None) or str(e)).strip()


def checks(d: Deps | None = None) -> list[str]:
    """Problems that stop a release (empty = go)."""
    d = d or Deps()
    problems: list[str] = []
    try:
        dirty = [l for l in d.git("status", "--porcelain").splitlines() if l.strip()]
        if dirty:
            problems.append("the working tree isn't clean: " + "; ".join(l.strip() for l in dirty[:5]))
    except (subprocess.CalledProcessError, OSError) as e:
        problems.append(f"git status failed: {_why(e)}")
    try:
        d.git("fetch", "origin", "main")
        head, main = d.git("rev-parse", "HEAD").strip(), d.git("rev-parse", "origin/main").strip()
        if head != main:
            problems.append(f"HEAD is not origin/main ({head[:7]} vs {main[:7]}): release from an "
                            "up-to-date main")
    except (subprocess.CalledProcessError, OSError) as e:
        problems.append(f"git couldn't compare HEAD with origin/main: {_why(e)}")
    bad = d.toolchain()
    if bad:
        problems.append("these tools aren't from the pixi environment: " + ", ".join(bad) +
                        "; release inside `pixi shell` (pipeline/README.md, Toolchain)")
    try:
        d.env_file()
    except ship.ShipError as e:
        problems.append(str(e))
    st = d.run(DVC_STATUS, cwd=config.PIPELINE_DIR)
    if st.returncode != 0 or (st.stdout or "").strip() not in ("", "{}"):
        problems.append("the data doesn't match dvc.lock; run `uv run sfzmaps build` and commit dvc.lock")
    try:
        problems += d.check_shipped(st.stdout if st.returncode == 0 else "{}")
    except ship.ShipError as e:
        problems.append(str(e))
    return problems


def _tally(script: str, out: str, s: Summary) -> None:
    """Fold a publish script's output into the summary."""
    if script == "publish-tiles.mjs":
        for m in re.finditer(r"^(?:would upload|upload) .*\((\d+) bytes\)$", out, re.M):
            s.files += 1
            s.bytes += int(m.group(1))
    else:
        m = re.search(r"to upload now: (\d+) file\(s\), (\d+) bytes; superseded: (\d+); "
                      r"expired to delete: (\d+)", out)
        if m:
            s.files += int(m.group(1))
            s.bytes += int(m.group(2))
            s.superseded += int(m.group(3))
            s.deleted += int(m.group(4))


SAFE = ("\nUploads already made are safe (content-hashed, never overwritten); a re-run skips them. "
        "The manifests were restored.")


def _restore(d: Deps) -> None:
    """Undo a half-done publish's manifest edits so a re-run starts from a clean tree."""
    try:
        d.git("restore", "--", *MANIFESTS)
    except (subprocess.CalledProcessError, OSError):
        pass


def release(dry_run: bool = False, no_pr: bool = False, d: Deps | None = None) -> Summary:
    d = d or Deps()
    problems = checks(d)
    if problems:
        raise ReleaseError("not releasing:\n  " + "\n  ".join(problems))
    d.echo("exporting the GIS downloads (unchanged files are skipped)")
    try:
        d.downloads(d.echo)
    except Exception as e:  # noqa: BLE001 - any export failure stops the release with its reason
        raise ReleaseError(f"the downloads export failed, nothing was published: {e}") from e
    s = Summary()
    for script in SCRIPTS:
        try:
            r = d.node(script, ["--dry-run"] if dry_run else [])
        except (ship.ShipError, OSError) as e:
            _restore(d)
            raise ReleaseError(f"{script} couldn't run: {e}{SAFE}")
        d.echo((r.stdout or "").rstrip())
        if r.returncode != 0:
            _restore(d)
            raise ReleaseError(f"{script} failed:\n{(r.stderr or r.stdout or '').strip()}{SAFE}")
        _tally(script, r.stdout or "", s)
    if dry_run:
        d.echo(f"dry run: would upload {s.files} file(s), {s.bytes:,} bytes; nothing published, "
               "committed or pushed")
        return s
    changed = [l for l in d.git("status", "--porcelain", "--", *MANIFESTS).splitlines() if l.strip()]
    if not changed:
        d.echo("nothing to release")
        return s
    try:
        sha = d.git("rev-parse", "--short=7", "HEAD").strip()
        before = d.git("rev-parse", "--abbrev-ref", "HEAD").strip()
    except (subprocess.CalledProcessError, OSError) as e:
        raise ReleaseError(f"git failed before the release commit ({_why(e)}); the manifests are "
                           f"changed but uncommitted: commit {' and '.join(MANIFESTS)} by hand")
    date = d.today()
    s.branch = f"release/{date}-{sha}"
    title = f"Release {date}"
    body = (f"Published to the Blob store: {s.files} file(s), {s.bytes:,} bytes; "
            f"{s.superseded} superseded, {s.deleted} expired file(s) deleted.\n\n"
            f"Manifests updated: {', '.join(MANIFESTS)}. Merging deploys.\n\n{PR_FOOTER}")
    step = "creating the branch"
    try:
        d.git("checkout", "-b", s.branch)
        step = "committing"
        d.git("add", "--", *MANIFESTS)
        d.git("commit", "-m", f"{title}\n\nUpdated manifests: {', '.join(MANIFESTS)}")
        if no_pr:
            d.echo(f"committed on {s.branch} (--no-pr: not pushed)")
        else:
            step = "pushing"
            d.git("push", "-u", "origin", s.branch)
            step = "opening the PR"
            r = d.run(["gh", "pr", "create", "--base", "main", "--title", title, "--body", body],
                      cwd=config.PIPELINE_DIR)
            if r.returncode != 0:
                raise subprocess.CalledProcessError(r.returncode, "gh", stderr=r.stderr)
            lines = (r.stdout or "").strip().splitlines()
            s.pr_url = lines[-1] if lines else None
    except (subprocess.CalledProcessError, OSError) as e:
        finish = {"creating the branch": f"git checkout -b {s.branch}, then commit the manifests",
                  "committing": f"commit the manifests on {s.branch}, then push and open the PR",
                  "pushing": f"git push -u origin {s.branch}, then gh pr create --base main --title '{title}'",
                  "opening the PR": f"gh pr create --base main --head {s.branch} --title '{title}'"}[step]
        raise ReleaseError(f"{step} failed ({_why(e)}). The release is published; the manifests are "
                           f"on branch {s.branch}. To finish: {finish}")
    finally:
        try:
            d.git("checkout", before)
        except (subprocess.CalledProcessError, OSError):
            pass
    d.echo(f"uploaded {s.files} file(s), {s.bytes:,} bytes; superseded {s.superseded}, "
           f"deleted {s.deleted}" + (f"; PR {s.pr_url}" if s.pr_url else ""))
    return s
