# One-Minute Test Run Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** One command runs every test (pipeline, site unit, end-to-end) in under 60 seconds of wall time on the owner's Mac.

**Architecture:** The three suites run at the same time, so each must finish in under 60 seconds. Make them faster by changing how they run, not by deleting or weakening tests.

**Owner's rule:** "A full test run should take no more than 1 minute".

## Baseline (measured)

- Pipeline: 74s on one core. Two `test_refine.py` fixtures (`big`, `shifted`) build near-identical synthetic sheets at about 8s each.
- Site unit tests: about 1s.
- End-to-end: about 6.5 minutes, with 1 worker, SwiftShader WebGL, a dev server, and real waits (the 8s hold limit and `waitForTimeout` fades). Tests over 10s account for 200s.

## Global Constraints

- Keep every test, and keep what it proves. Weakening an assertion or skipping a test to hit the time is not allowed. Deleting tests is a separate decision.
- Keep `site/scripts/check-tiles-env.mjs` working, and keep the e2e published-tiles mode.
- Production code changes only where a test needs a hook, behind test-only flags as `NEXT_PUBLIC_E2E` does now.
- Report measured wall times, best of 3 runs.

---

### Task 1: Pipeline under 30s

- [ ] Add `pytest-xdist` as a dev dependency, and run with `-n auto` by default in `pyproject.toml`.
- [ ] Build the `big` and `shifted` synthetic sheets once per session (cached, with `scope="session"`), or derive one from the other, keeping every assertion.
- [ ] Fix tests that share files or ports under parallel runs, if any fail.
- [ ] Measure: `time uv run pytest`, expected under 30s with all passing. Commit.

### Task 2: End-to-end under 60s

- [ ] Profile: list each test's time and what it waits on.
- [ ] Run against a production build (`next build && next start`) instead of the dev server, if it saves compile time. Keep a reused server.
- [ ] Replace real waits with Playwright's `page.clock`, or with lowering the hold limit and fade durations through an E2E-only setting, while asserting the same sequence of states.
- [ ] Run several workers. GPU WebGL (`--use-angle=metal`) is available on the Mac; with workers starving SwiftShader before, measure 2, 4 and 6 workers and pick the fastest that stays deterministic over 3 full runs.
- [ ] Measure: `time npm run test:e2e`, expected under 60s, all passing, 3 runs in a row. Commit.

### Task 3: One command

- [ ] Add a root `scripts/test-all.sh` that starts the three suites in parallel, prints each one's time and result, and exits non-zero if any fail. Update `AGENTS.md` and `CONTRIBUTING.md` to use it, with the e2e step needing no size rule any more.
- [ ] Measure: `time scripts/test-all.sh` under 60s. Commit, open a PR.
