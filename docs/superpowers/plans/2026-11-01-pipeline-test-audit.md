# Pipeline Test Audit Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A verdict on every test in `pipeline/tests`, with evidence from reading it and measuring it, so the owner can decide what to delete.

**Architecture:** Measure first: per-test time and the lines only that test covers. Then one fresh reader per group of files reads every test and the code it covers. Then a script checks completeness, and a skeptic tries to overturn every delete.

**Spec:** None. The owner's rules: "no trivial assertions", no testing third-party libraries, test our behaviour only; "You are wasting so much time and electricity of useless tests"; and, on the first audit, "Do a proper audit. Why are you being lazy?"

## Global Constraints

- Read-only on the repo: no test or code changes. Nothing is deleted in this plan.
- All output goes to `$SCRATCH/test-audit/` (the session scratchpad). Coverage tools are added only per run with `uv run --with`, never to the project.
- No API calls. Never read env files. Tests already mock paid calls; stop if one doesn't.
- A verdict must come from reading the test's body and the code it covers. Test names alone are not evidence, and neither is the first audit.

## Verdicts

| Verdict | Meaning |
|---|---|
| KEEP | A failure would mean real harm the owner wouldn't otherwise catch: paid spending, secrets, lost human edits, corrupted or wrongly published data, tricky logic such as geometry, scoring, parsing |
| DELETE | Restated constant, trivial getter, mock behaviour, third-party behaviour, wording or formatting, or a duplicate of a named better test |
| MERGE | Several tests that could be one parametrized test |
| FIX | Protects something real but asserts too weakly to catch it |

## Review Focus

- A test that looks trivial but is the only cover for spending or secrets code: its unique-lines count must be weighed.
- A "duplicate" whose named twin doesn't really cover the same case.
- A slow test with high value: report it as slow, and don't delete it because of that.
- Parametrized tests: judge per function, and report the case count.
- Tests guarding a past real bug: look at `git log -S` on anything unclear.

---

### Task 1: Measure

- [ ] Run the suite with `--durations=0`, JUnit XML and per-test coverage contexts: `uv run --with pytest-cov pytest --cov=sfzmaps --cov-context=test --cov-report= --junitxml=$SCRATCH/test-audit/junit.xml`. Expected: all pass.
- [ ] Produce `measures.csv`: file, test function, cases, seconds, lines covered, lines only this test covers, where they are.
- [ ] Report suite total, slowest 20, and time per file.

### Task 2: Read every test, in parallel

One fresh reader per group, given `measures.csv`:

1. `test_editor_api`
2. `test_refine`, `test_runner`
3. `test_llm`, `test_cli`, `test_config`, `test_manifest`
4. `test_share`, `test_drive`, `test_datasf`, `test_pointsfile`, `test_tiles`
5. `test_warp`, `test_transform`, `test_stitch`, `test_placement`, `test_anchors`
6. `test_linework`, `test_score`, `test_evaluate`
7. `test_pages`, `test_reading`, `test_crossref`, `test_reviewpage`
8. `test_districts_assemble`, `_rules`, `_shared`, `_today`, `_validate`
9. `test_districts_labels`, `_pattern`, `_regionread`, `_sampling`

- [ ] Each reader opens each test file in full and the source functions each test calls, and writes `verdicts/<group>.csv`: `file,test,verdict,protects,evidence,twin`. `evidence` cites source `file:line` and what the test asserts. `twin` names the better test for a duplicate.

### Task 3: Check and challenge

- [ ] A script compares verdicts to collected tests. Expected: every function has one row, with no empty evidence and no DELETE-as-duplicate lacking a twin that exists.
- [ ] A skeptic reads every DELETE that covers unique lines or touches spending, secrets, review edits, publishing or warping, and tries to overturn it. Overturned rows become KEEP, with the reason recorded.

### Task 4: Report

- [ ] Write `report.md`: totals by verdict, time saved measured from `measures.csv`, what risks stay covered, every test by file, and lines that lose all cover. Give it to the owner to decide.
