"""Recover a file this session wrote and later lost, from the session transcript: replay every Write (full content)
and Edit (old -> new) tool call on that path, in order, up to an optional cut-off time.  Bash commands that also name
the path and look like in-place edits (python heredoc patches, sed -i, perl -pi, > redirects) are listed, not
applied: check them by hand.
usage: recover_from_transcript.py TRANSCRIPT.jsonl PATH [--until 2026-11-09T11:30] [--out FILE] [--events]"""
import argparse
import json
import re
import sys


def events(transcript, path):
    seen = set()
    for n, line in enumerate(open(transcript)):
        if path not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        m = d.get("message", {})
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        ts = d.get("timestamp", "")
        for x in m["content"]:
            if x.get("type") != "tool_use":
                continue
            inp = x.get("input", {})
            key = x.get("id")
            if key in seen:
                continue
            seen.add(key)
            if inp.get("file_path") == path and "content" in inp:
                yield n, ts, "write", inp
            elif inp.get("file_path") == path and ("old_string" in inp or "old_str" in inp):
                yield n, ts, "edit", inp
            elif path in json.dumps(inp) and "command" in inp and re.search(
                    r"open\([^)]*['\"]w|sed -i|perl -pi|>\s*\S*" + re.escape(path.split("/")[-1]) + r"|write_text|\.replace\(",
                    inp["command"]):
                yield n, ts, "bash", inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("path")
    ap.add_argument("--until", default="")
    ap.add_argument("--out")
    ap.add_argument("--events", action="store_true", help="list the events only")
    a = ap.parse_args()
    text, n_ok, bash = None, 0, []
    for n, ts, kind, inp in events(a.transcript, a.path):
        if a.until and ts and ts > a.until:
            break
        if a.events:
            print(f"{n:7d} {ts[:19]} {kind}", (inp.get("description") or "")[:90])
        if kind == "write":
            text, n_ok = inp["content"], n_ok + 1
        elif kind == "edit" and text is not None:
            old = inp.get("old_string", inp.get("old_str"))
            new = inp.get("new_string", inp.get("new_str"))
            if old in text:
                text = text.replace(old, new) if inp.get("replace_all") else text.replace(old, new, 1)
                n_ok += 1
            else:
                print(f"  ! edit at line {n} did not apply", file=sys.stderr)
        elif kind == "bash":
            bash.append((n, ts, inp["command"][:300]))
    if text is None:
        sys.exit(f"no Write of {a.path} found")
    if a.out:
        open(a.out, "w").write(text)
    print(f"{a.path}: {n_ok} writes/edits applied; {len(bash)} shell commands that may also edit it:", file=sys.stderr)
    for n, ts, c in bash:
        print(f"  line {n} {ts[:19]}: {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
