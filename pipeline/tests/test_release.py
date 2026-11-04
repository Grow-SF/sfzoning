import subprocess

import pytest

from sfzmaps import release

SHA = "a" * 40


class Fake:
    """Injected git / node / gh / dvc; records calls."""
    def __init__(self, dirty="", head=SHA, main=SHA, shipped=(), changed=" M site/lib/downloads.json",
                 dvc=(0, "{}")):
        self.dirty, self.head, self.main, self.shipped, self.changed, self.dvc = dirty, head, main, shipped, changed, dvc
        self.git_calls, self.node_calls, self.run_calls, self.out, self.downloads_calls = [], [], [], [], 0

    def git(self, *a):
        self.git_calls.append(a)
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
        if cmd[0] == "gh":
            return subprocess.CompletedProcess(cmd, 0, "https://github.com/x/y/pull/9\n", "")
        return subprocess.CompletedProcess(cmd, self.dvc[0], self.dvc[1], "")

    def deps(self):
        def dl():
            self.downloads_calls += 1
            return {}
        return release.Deps(git=self.git, node=self.node, run=self.run, check_shipped=lambda s: list(self.shipped),
                            downloads=dl, today=lambda: "2026-11-04", echo=self.out.append)

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
