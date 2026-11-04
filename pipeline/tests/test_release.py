import subprocess

import pytest

from sfzmaps import release, ship

SHA = "a" * 40


class Fake:
    """Injected git / node / gh / dvc; records calls."""
    def __init__(self, dirty="", head=SHA, main=SHA, shipped=(), changed=" M site/lib/downloads.json",
                 dvc=(0, "{}"), git_fail=None, gh=None, env_ok=True):
        self.git_fail, self.gh, self.env_ok = git_fail or {}, gh, env_ok
        self.dirty, self.head, self.main, self.shipped, self.changed, self.dvc = dirty, head, main, shipped, changed, dvc
        self.git_calls, self.node_calls, self.run_calls, self.out, self.downloads_calls = [], [], [], [], 0

    def git(self, *a):
        self.git_calls.append(a)
        if a[0] in self.git_fail:
            raise subprocess.CalledProcessError(1, ["git", *a], stderr=self.git_fail[a[0]])
        if a[:2] == ("status", "--porcelain"):
            return self.changed if "--" in a else self.dirty
        if a[:2] == ("rev-parse", "HEAD"):
            return self.head
        if a[:2] == ("rev-parse", "origin/main"):
            return self.main
        if a[:2] == ("rev-parse", "--short=7"):
            return self.head[:7]
        if a[:2] == ("rev-parse", "--abbrev-ref"):
            return "main"
        return ""

    def node(self, script, args):
        self.node_calls.append((script, list(args)))
        out = ("to upload now: 2 file(s), 300 bytes; superseded: 1; expired to delete: 1\n"
               if "downloads" in script else "upload tiles/a.pmtiles -> x.pmtiles (700 bytes)\n")
        return subprocess.CompletedProcess([], 0, out, "")

    def run(self, cmd, cwd=None):
        self.run_calls.append(cmd)
        if cmd[0] == "gh" and self.gh == "missing":
            raise FileNotFoundError("gh")
        if cmd[0] == "gh":
            return subprocess.CompletedProcess(cmd, 0, "https://github.com/x/y/pull/9\n", "")
        return subprocess.CompletedProcess(cmd, self.dvc[0], self.dvc[1], "")

    def deps(self):
        def dl():
            self.downloads_calls += 1
            return {}
        return release.Deps(git=self.git, node=self.node, run=self.run, check_shipped=lambda s: list(self.shipped),
                            downloads=dl, env_file=self._env, today=lambda: "2026-11-04", echo=self.out.append)

    def _env(self):
        if not self.env_ok:
            raise ship.ShipError("site/.env.local is missing")

    def pushed(self):
        return any(c[0] == "push" for c in self.git_calls)


def test_release_stops_on_dirty_tree():
    f = Fake(dirty=" M pipeline/x.py")
    with pytest.raises(release.ReleaseError, match="isn't clean"):
        release.release(d=f.deps())
    assert f.node_calls == [] and f.downloads_calls == 0


def test_release_stops_when_not_at_origin_main():
    f = Fake(head="b" * 40)
    with pytest.raises(release.ReleaseError, match="HEAD is not origin/main"):
        release.release(d=f.deps())
    assert ("fetch", "origin", "main") in f.git_calls
    assert f.node_calls == []


def test_release_stops_when_cache_not_shipped():
    f = Fake(shipped=["answer pack abc wasn't confirmed"])
    with pytest.raises(release.ReleaseError, match="wasn't confirmed"):
        release.release(d=f.deps())
    assert f.node_calls == [] and f.downloads_calls == 0


def test_release_stops_when_dvc_data_differs():
    f = Fake(dvc=(0, '{"stage": ["changed deps"]}'))
    with pytest.raises(release.ReleaseError, match="dvc.lock"):
        release.release(d=f.deps())


def test_dry_run_passes_dry_run_to_both_scripts_and_commits_nothing():
    f = Fake()
    release.release(dry_run=True, d=f.deps())
    assert f.node_calls == [("publish-tiles.mjs", ["--dry-run"]), ("publish-downloads.mjs", ["--dry-run"])]
    assert f.downloads_calls == 1
    assert not any(c[0] in ("checkout", "add", "commit", "push") for c in f.git_calls)
    assert not any(c[0] == "gh" for c in f.run_calls)


def test_no_changes_makes_no_branch():
    f = Fake(changed="")
    s = release.release(d=f.deps())
    assert "nothing to release" in f.out
    assert not any(c[0] in ("checkout", "commit", "push") for c in f.git_calls)
    assert s.branch is None


def test_release_commits_branch_pushes_and_opens_pr():
    f = Fake()
    s = release.release(d=f.deps())
    assert s.branch == "release/2026-11-04-aaaaaaa"
    assert ("checkout", "-b", s.branch) in f.git_calls
    assert f.pushed()
    gh = next(c for c in f.run_calls if c[0] == "gh")
    assert gh[gh.index("--base") + 1] == "main"
    assert gh[gh.index("--title") + 1] == "Release 2026-11-04"
    assert gh[gh.index("--body") + 1].endswith("🤖 Generated with [Claude Code](https://claude.com/claude-code)")
    assert (s.files, s.bytes, s.superseded, s.deleted) == (3, 1000, 1, 1)
    assert s.pr_url == "https://github.com/x/y/pull/9"


def test_no_pr_commits_but_does_not_push():
    f = Fake()
    s = release.release(no_pr=True, d=f.deps())
    assert any(c[0] == "commit" for c in f.git_calls)
    assert not f.pushed() and not any(c[0] == "gh" for c in f.run_calls)
    assert s.pr_url is None


def test_script_failure_stops_before_commit():
    f = Fake()
    d = f.deps()
    d.node = lambda script, args: subprocess.CompletedProcess([], 1, "", "boom")
    with pytest.raises(release.ReleaseError, match="boom"):
        release.release(d=d)
    assert not any(c[0] == "commit" for c in f.git_calls)


def test_second_script_failing_restores_manifests_and_says_uploads_are_safe():
    f = Fake()
    d = f.deps()
    def node(script, args):
        f.node_calls.append((script, list(args)))
        if "downloads" in script:
            return subprocess.CompletedProcess([], 1, "", "blob down")
        return subprocess.CompletedProcess([], 0, "upload t (5 bytes)\n", "")
    d.node = node
    with pytest.raises(release.ReleaseError, match="never overwritten") as e:
        release.release(d=d)
    assert "blob down" in str(e.value)
    assert ("restore", "--", *release.MANIFESTS) in f.git_calls
    assert not any(c[0] in ("checkout", "commit", "push") for c in f.git_calls)


def test_push_failure_returns_to_start_branch_with_how_to_finish():
    f = Fake(git_fail={"push": "denied"})
    with pytest.raises(release.ReleaseError) as e:
        release.release(d=f.deps())
    msg = str(e.value)
    assert "denied" in msg and "git push -u origin release/2026-11-04-aaaaaaa" in msg
    assert f.git_calls[-1] == ("checkout", "main")


def test_missing_gh_is_a_release_error_and_returns_to_start_branch():
    f = Fake(gh="missing")
    with pytest.raises(release.ReleaseError, match="gh pr create --base main --head release/"):
        release.release(d=f.deps())
    assert f.git_calls[-1] == ("checkout", "main")


def test_missing_env_file_stops_before_the_export():
    f = Fake(env_ok=False)
    with pytest.raises(release.ReleaseError, match="env.local is missing"):
        release.release(d=f.deps())
    assert f.downloads_calls == 0


def test_git_status_failure_is_a_problem_not_a_traceback():
    f = Fake(git_fail={"status": "not a repo"})
    with pytest.raises(release.ReleaseError, match="git status failed"):
        release.release(d=f.deps())


def test_tally_reads_the_scripts_real_output_lines():
    tiles = ("same     a.pmtiles\n"
             "upload b.pmtiles -> b-1234abcd.pmtiles (700 bytes)\n"
             "would upload c.pmtiles -> c-9999.pmtiles (50 bytes)\n")
    dl = ("listing: 10 names under downloads/ and dvc/files/md5/ in 1 request(s)\n"
          "files: 4 (2 new uploads planned), sheets: 9 (3 to upload as packed)\n"
          "to upload now: 5 file(s), 12345 bytes; superseded: 2; expired to delete: 1\n"
          "requests: 7 listing and upload, 1 delete\n")
    s = release.Summary()
    release._tally("publish-tiles.mjs", tiles, s)
    release._tally("publish-downloads.mjs", dl, s)
    assert (s.files, s.bytes, s.superseded, s.deleted) == (7, 13095, 2, 1)
