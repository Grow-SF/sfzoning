import json
import math
from pathlib import Path

import cv2
import typer

from sfzmaps import config
from sfzmaps.sources import drive

app = typer.Typer(no_args_is_help=True, help="Georeference historic SF zoning maps.")


@app.callback()
def main() -> None:
    """Georeference historic SF zoning maps."""


@app.command()
def fetch(
    only: str = typer.Option(
        None, "--only", help="Only fetch this top-level subfolder (by Drive title)."
    ),
) -> None:
    """Mirror the Drive tree into data/raw/."""
    root = drive.list_folder(config.ROOT_FOLDER_ID)
    folders = [e for e in root if e.is_folder]
    if only is not None:
        folders = [e for e in folders if e.title == only]
        if not folders:
            raise typer.BadParameter(f"no top-level folder titled {only!r}")
    for e in root:
        if not e.is_folder and only is None:
            config.RAW_DIR.mkdir(parents=True, exist_ok=True)
            p = config.RAW_DIR / drive._safe_name(e.title)
            if p.exists() and drive.looks_like_html(p):  # a saved Drive error page
                p.unlink()
            if not (p.exists() and p.stat().st_size > 0):
                drive.download(e.id, p)
    n = 0
    for e in folders:
        typer.echo(f"fetching {e.title}")
        n += len(drive.sync(e.id, config.RAW_DIR / drive._safe_name(e.title)))
    typer.echo(f"{n} files in {config.RAW_DIR}")


@app.command()
def refdata(
    force: bool = typer.Option(False, "--force", help="Re-download even if cached."),
) -> None:
    """Download DataSF parcels/streets and build blocks, corners, intersections."""
    from sfzmaps.sources import datasf

    if force or not (config.REF_DIR / "parcels.parquet").exists() \
            or not (config.REF_DIR / "streets.parquet").exists():
        datasf.fetch_reference(config.REF_DIR)
    ref = datasf.load_reference()
    typer.echo(f"blocks: {len(ref.blocks)}")
    typer.echo(f"corners: {len(ref.corners)}")
    typer.echo(f"intersections: {len(ref.intersections)}")
    typer.echo(f"bounds: {list(ref.blocks.total_bounds)}")


def _page_png(page) -> Path:
    from sfzmaps import pages

    return pages.render(page, config.WORK_DIR / page.series / page.sheet_id / "page.png")


@app.command()
def calibrate(
    year: str = typer.Option(..., "--year", help="Series, e.g. 1948."),
    sheet: str = typer.Option(..., "--sheet", help="Sheet id of one representative map sheet."),
) -> None:
    """Find the smallest label-reading scale that keeps >= 95% of full-resolution recall."""
    from sfzmaps import llm, pages, reading

    match = [p for p in pages.discover() if p.series == year and p.sheet_id == sheet]
    if not match:
        raise typer.BadParameter(f"no page {year}/{sheet}")
    rep: dict[float, float] = {}
    try:
        best = reading.calibrate_scale(_page_png(match[0]), report=rep)
    except (llm.LLMAuthError, llm.LLMRefusal, RuntimeError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    for s, r in rep.items():
        typer.echo(f"scale {s}: recall {r:.1%}")
    config.save_label_scale(year, best)
    typer.echo(f"saved label scale {best} for {year} to {config.WORK_DIR / 'label_scale.json'}")


@app.command()
def estimate(
    year: str = typer.Option(None, "--year", help="Only this series."),
    all_: bool = typer.Option(False, "--all", help="All series."),
) -> None:
    """Project API calls, image tokens and USD for classify + label reading (no API calls)."""
    from sfzmaps import llm, pages, reading

    if not (year or all_):
        raise typer.BadParameter("pass --year Y or --all")
    in_price, out_price, out_tokens = config.PRICE_IN_PER_M / 1e6, config.PRICE_OUT_PER_M / 1e6, config.EST_OUT_TOKENS
    calls = tokens = hits = npages = fallback = 0
    for page in pages.discover():
        if year and page.series != year:
            continue
        npages += 1
        png_path = _page_png(page)
        png = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        thumb, _ = reading.thumbnail(png)
        jobs = [thumb]
        # Use mask boxes when the classification is already cached (no call); else neatline only.
        info = None
        if llm.is_cached(thumb, reading.classify_prompt(thumb), reading.PageInfo):
            info = reading.classify_page(png_path)
        else:
            fallback += 1
        skip = reading.exclusion_mask(png, info)
        _, tiles = reading.select_tiles(png, scale=config.label_scale(page.series), skip=skip)
        jobs += [t for _, _, t in tiles]
        for i, img in enumerate(jobs):
            prompt, schema = (reading.classify_prompt(img), reading.PageInfo) if i == 0 \
                else (reading.label_prompt(img), reading._Labels)
            if llm.is_cached(img, prompt, schema):
                hits += 1
                continue
            calls += 1
            tokens += config.estimate_image_tokens(img.shape[1], img.shape[0])
    text_tokens = calls * config.EST_TEXT_TOKENS_PER_CALL
    usd = (tokens + text_tokens) * in_price + calls * out_tokens * out_price
    typer.echo(f"pages: {npages}")
    typer.echo(f"calls: {calls} (cache hits: {hits})")
    typer.echo(f"image tokens: {tokens:,.0f}")
    typer.echo(f"prompt text tokens: {text_tokens:,.0f} ({config.EST_TEXT_TOKENS_PER_CALL}/call)")
    typer.echo(f"projected cost (estimate): ${usd:,.2f}  (${config.PRICE_IN_PER_M:g}/M in, ${config.PRICE_OUT_PER_M:g}/M out, {out_tokens} output tokens/call)")
    if fallback:
        typer.echo(f"note: {fallback} of {npages} pages have no cached classification, so mask boxes "
                   "are unknown and their tiles are excluded by neatline only (upper bound).")


@app.command()
def run(
    year: str = typer.Option(None, "--year", help="Series, e.g. 1948."),
    all_: bool = typer.Option(False, "--all", help="All series."),
    sheet: str = typer.Option(None, "--sheet", help="Only this sheet id."),
    until: str = typer.Option(None, "--until", help="Stop after this stage."),
    force_auto: bool = typer.Option(False, "--force-auto", help="Ignore review.json for this run."),
    force: bool = typer.Option(False, "--force", help="Rerun every stage (config changes do not trigger reruns)."),
) -> None:
    """Run the pipeline per sheet (skips fresh stages; human reviews are kept)."""
    import logging

    from sfzmaps import pages, runner
    from sfzmaps.sources import datasf

    if not (year or all_):
        raise typer.BadParameter("pass --year Y or --all")
    if until is not None and until not in runner.STAGES:
        raise typer.BadParameter(f"--until must be one of {', '.join(runner.STAGES)}")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ref = datasf.load_reference()
    allp = pages.discover()
    from sfzmaps import llm

    failures: list[tuple[str, str]] = []
    todo = sorted({p.series for p in allp if all_ or p.series == year})
    for k, s in enumerate(todo):
        try:
            runner.run_series(s, sheet=sheet, until=until, force_auto=force_auto, force=force,
                              ref=ref, pages_list=allp)
            if sheet is None and until is None:  # a partial run must not republish the series
                _publish(s)
        except llm.FATAL as e:  # quota / auth: every later call fails too; stop the run
            _resume_hint(e, "run", todo[k:], all_)
            raise typer.Exit(2)
        except Exception as e:  # noqa: BLE001 - one failing series must not stop the others
            logging.getLogger(__name__).exception("series %s failed", s)
            failures.append((s, f"{type(e).__name__}: {e}"))
    if failures:
        typer.echo(f"{len(failures)} series failed:", err=True)
        for s, msg in failures:
            typer.echo(f"  {s}: {msg[:300]}", err=True)
        raise typer.Exit(1)


def _resume_hint(e: Exception, command: str, remaining: list[str], all_: bool) -> None:
    """Explain a fatal vision error (quota / spend limit / auth) and how to resume."""
    flag = "--year" if command == "run" else "--series"
    typer.echo(f"stopped: {type(e).__name__}: {e}", err=True)
    typer.echo(f"not finished: {', '.join(remaining)}", err=True)
    how = f"sfzmaps {command} --all" if all_ else \
        " ; ".join(f"sfzmaps {command} {flag} {s}" for s in remaining)
    typer.echo(f"resume when the API is usable again with: {how}  (answers already read are "
               "cached and not paid for twice)", err=True)


def _publish(series: str, min_zoom: int = 11, max_zoom: int = 18) -> None:
    """Stitch -> tiles -> manifest for one series (scores read from each sheet's score.json)."""
    from sfzmaps import manifest, reviewpage, stitch, tiles

    scores = stitch.load_scores(series)
    vrt = stitch.build_vrt(series, scores)
    out = config.OUT_DIR / series / f"{series}.pmtiles"
    report = tiles.make_pmtiles(vrt, out, min_zoom=min_zoom, max_zoom=max_zoom,
                                name=f"SF zoning {series}",
                                attribution=config.RUMSEY_CREDIT if series == "1948" else None)
    path = manifest.write(series, scores, report)
    typer.echo(f"{series}: {out} ({report.bytes_total / 1e6:.1f} MB of tiles); manifest {path}")
    for z, b in sorted(report.bytes_by_zoom.items()):
        typer.echo(f"  z{z}: {b / 1e6:.2f} MB")
    try:
        typer.echo(f"{series}: review page {reviewpage.write(series)}")
    except Exception as e:  # noqa: BLE001 - the review page is optional (needs GDAL/parcels)
        typer.echo(f"{series}: review page skipped ({type(e).__name__}: {e})", err=True)


@app.command()
def tiles(
    year: str = typer.Option(..., "--year", help="Series, e.g. 1948."),
    min_zoom: int = typer.Option(11, "--min-zoom"),
    max_zoom: int = typer.Option(18, "--max-zoom"),
) -> None:
    """(Re)build the stitched VRT, PMTiles and manifest for one series from existing warps."""
    try:
        _publish(year, min_zoom, max_zoom)
    except Exception as e:  # noqa: BLE001 - gdal/pmtiles subprocess errors, missing tools, ...
        typer.echo(f"error: {type(e).__name__}: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def paper(
    year: str = typer.Option(None, "--year", help="Series, e.g. 1948."),
    all_: bool = typer.Option(False, "--all", help="Every series with warped sheets."),
    min_zoom: int = typer.Option(11, "--min-zoom"),
    max_zoom: int = typer.Option(18, "--max-zoom"),
    force: bool = typer.Option(False, "--force", help="Re-warp sheets even when fresh."),
) -> None:
    """Warp every placed sheet as it looks on paper (opaque, untouched RGB) and publish
    out/<series>/<series>-paper.pmtiles, the layer the site shows as "the map as drawn"."""
    from sfzmaps import reading, runner, stitch, tiles as tiles_mod, warp
    from sfzmaps.transform import SheetTransform

    if not year and not all_:
        raise typer.BadParameter("pass --year or --all")
    series_list = sorted(p.name for p in config.WORK_DIR.iterdir() if p.is_dir()) if all_ else [year]
    for series in series_list:
        n = 0
        for d in sorted(p for p in (config.WORK_DIR / series).iterdir() if p.is_dir()):
            tj, png, pj = d / "transform.json", d / "page.png", d / "page.json"
            if not (tj.exists() and png.exists() and pj.exists() and (d / "warped.tif").exists()):
                continue
            out = d / stitch.raster_name(True)
            srcs = [tj, png, pj, Path(warp.__file__)]
            if not force and out.exists() and all(out.stat().st_mtime_ns >= p.stat().st_mtime_ns for p in srcs):
                continue
            runner._warp_checked(png, SheetTransform.from_json(json.loads(tj.read_text())),
                                 reading.load_page(pj).mask_boxes, out, paper=True)
            n += 1
        try:
            vrt = stitch.build_vrt(series, paper=True)
        except RuntimeError as e:
            typer.echo(f"{series}: skipped ({e})", err=True)
            continue
        out = config.OUT_DIR / series / f"{series}-paper.pmtiles"
        report = tiles_mod.make_pmtiles(vrt, out, min_zoom=min_zoom, max_zoom=max_zoom, quality=90,
                                        name=f"SF zoning {series} (paper)",
                                        attribution=config.RUMSEY_CREDIT if series == "1948" else None)
        typer.echo(f"{series}: {n} sheets warped; {out} ({report.bytes_total / 1e6:.1f} MB)")


@app.command()
def districts(
    series: str = typer.Option(None, "--series", help="Series (e.g. 1948, 1960-zoning) or `today`."),
    all_: bool = typer.Option(False, "--all", help="Every district series, plus today."),
    sheet: list[str] = typer.Option(None, "--sheet", help="Only these sheet ids (testing): writes a "
                                                         "partial artifact (districts-partial/), "
                                                         "never the full-series outputs."),
    validate: bool = typer.Option(False, "--validate", help="2000-digital-zoning vs DataSF today."),
    republish: bool = typer.Option(False, "--republish", help="Re-apply overrides, height and rules "
                                                             "to the saved reads (no vision calls)."),
    cache_only: bool = typer.Option(False, "--cache-only", help="Vision answers from the disk cache "
                                                               "only: region reads fall back to "
                                                               "inheritance, other misses fail."),
    status: bool = typer.Option(False, "--status", help="Provenance of every series' reads and "
                                                       "which need rerunning."),
) -> None:
    """Read zoning districts per parcel, join the rules, publish out/<series>/districts/."""
    import json
    import logging

    import pandas as pd

    from sfzmaps import llm, reviewpage
    from sfzmaps.districts import assemble
    from sfzmaps.districts.series import SERIES

    if status:
        typer.echo(assemble.format_status(assemble.status_rows()))
        if not (series or all_ or validate):
            return
    if not (series or all_ or validate):
        raise typer.BadParameter("pass --series S, --all, --validate or --status")
    if series and series != "today" and SERIES.get(series, None) is None:
        raise typer.BadParameter(f"unknown series {series!r}")
    if series and series != "today" and SERIES[series].role == "skip":
        raise typer.BadParameter(f"{series}: special-use series are out of scope")
    if republish and sheet:
        raise typer.BadParameter("--republish works on whole series; drop --sheet")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if republish:
        todo = [s for s, i in SERIES.items() if i.role == "use"] + ["today"] if all_ else \
            ([series] if series else [])
    else:
        todo = [s for s, i in SERIES.items() if i.role != "skip"] + ["today"] if all_ else \
            ([series] if series else [])
    if republish or cache_only:
        llm.CACHE_ONLY = True       # nothing below may reach the network
    live0, in0, out0 = llm.CallCounter.live, llm.CallCounter.in_tokens, llm.CallCounter.out_tokens
    hits0 = llm.CallCounter.tile_hits
    failures: list[tuple[str, str]] = []
    for k, s in enumerate(todo):
        try:
            if republish:
                d = assemble.republish(s)
            else:
                d = assemble.run_series(s, sheets=sheet or None, cache_only=cache_only)
            st = d / "stats.json"
            if st.exists():
                j = json.loads(st.read_text())
                typer.echo(f"{s}: {d} (unknown share {j['unknown_share']:.1%})")
            else:
                typer.echo(f"{s}: {d}")
            page = s if s == "today" or SERIES[s].role != "height" else assemble.zoning_series_of(s)
            if page != "today" and not sheet:
                try:
                    typer.echo(f"{page}: review page {reviewpage.write(page)}")
                except Exception as e:  # noqa: BLE001 - the review page is optional
                    typer.echo(f"{page}: review page skipped ({type(e).__name__}: {e})", err=True)
        except llm.FATAL as e:  # quota / spend limit / auth: every later call fails too
            _resume_hint(e, "districts", todo[k:], all_)
            raise typer.Exit(2)
        except Exception as e:  # noqa: BLE001 - one failing series must not stop the others
            logging.getLogger(__name__).exception("series %s failed", s)
            failures.append((s, f"{type(e).__name__}: {e}"))
    if todo:
        tin, tout = llm.CallCounter.in_tokens - in0, llm.CallCounter.out_tokens - out0
        usd = tin * config.PRICE_IN_PER_M / 1e6 + tout * config.PRICE_OUT_PER_M / 1e6
        typer.echo(f"live vision calls: {llm.CallCounter.live - live0} "
                   f"({tin:,} in / {tout:,} out tokens, ~${usd:.2f})")
        if llm.CallCounter.tile_hits - hits0 > 0:
            typer.echo(f"tile cache: {llm.CallCounter.tile_hits - hits0} tiles reused")
    if validate:
        try:
            from sfzmaps.districts import validate as val
        except ImportError:
            typer.echo("error: validation is not implemented yet (districts/validate.py)", err=True)
            raise typer.Exit(1)
        import geopandas as gpd

        def table(s):
            return pd.DataFrame(gpd.read_file(assemble.out_dir(s) / "parcels.fgb").drop(columns="geometry"))

        a, b = table("2000-digital-zoning"), table("today")
        typer.echo(json.dumps(val.agreement(a, b), indent=1, default=str))
        keys = config.PIPELINE_DIR / "tests" / "fixtures" / "district_keys"
        typer.echo("series                 parcels  unknown  key_accuracy")
        for s in [s for s, i in SERIES.items() if i.role == "use"] + ["today"]:
            st = assemble.out_dir(s) / "stats.json"
            if not st.exists():
                continue
            j = json.loads(st.read_text())
            n = sum(j[c]["parcels"] for c in assemble.CATEGORIES)
            kf = keys / f"{s}.csv"
            acc = f"{val.key_accuracy(table(s), kf):.1%}" if kf.exists() else "-"
            typer.echo(f"{s:22} {n:8,} {j['unknown_share']:8.1%}  {acc}")
    if failures:
        typer.echo(f"{len(failures)} series failed:", err=True)
        for s, msg in failures:
            typer.echo(f"  {s}: {msg[:300]}", err=True)
        raise typer.Exit(1)


@app.command()
def crossref(
    year: str = typer.Option(None, "--year", help="Series, e.g. 1972."),
    all_: bool = typer.Option(False, "--all", help="Every series with flagged sheets."),
    sheet: str = typer.Option(None, "--sheet", help="Only this sheet id."),
    max_refs: int = typer.Option(4, "--max-refs", help="Passing counterparts to try per sheet."),
    publish: bool = typer.Option(True, "--publish/--no-publish", help="Republish improved series."),
) -> None:
    """Re-place flagged sheets by matching them to the same sheet in another edition that
    passed (sfzmaps.crossref); keep a result only if it scores better. No API calls."""
    import logging
    import shutil

    from sfzmaps import crossref as xr, manifest, pages, reading, runner, stitch
    from sfzmaps.sources import datasf

    if not (year or all_):
        raise typer.BadParameter("pass --year Y or --all")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ref = datasf.load_reference()
    allp = pages.discover()
    by_key = {(p.series, p.sheet_id): p for p in allp}
    numbers: dict[tuple[str, str], str | None] = {}
    passed: set[tuple[str, str]] = set()
    flagged: dict[str, list[str]] = {}
    for d in sorted(config.WORK_DIR.glob("*/*/page.json")):
        series, sid = d.parent.parent.name, d.parent.name
        info = reading.load_page(d)
        if info.kind != "map":
            continue
        numbers[(series, sid)] = info.sheet_number
    for series in sorted({s for s, _ in numbers}):
        for sid, sc in stitch.load_scores(series).items():
            st = manifest.status_of(series, sid, sc)
            if st != "needs_review":
                if (config.WORK_DIR / series / sid / "transform.json").exists():
                    passed.add((series, sid))
            elif not str(sc.reason or "").startswith(("skipped", "error")) \
                    and not (config.WORK_DIR / series / sid / "review.json").exists():
                flagged.setdefault(series, []).append(sid)
    todo = sorted(flagged) if all_ else [year]
    summary: list[str] = []

    def say(line: str) -> None:   # each sheet's outcome as soon as it's known
        summary.append(line)
        typer.echo(line)
    for series in todo:
        kept = 0
        for sid in flagged.get(series, []):
            if sheet and sid != sheet:
                continue
            d = config.WORK_DIR / series / sid
            old = stitch.load_scores(series).get(sid)
            cands = xr.candidates(series, numbers.get((series, sid)), passed, numbers)[:max_refs]
            best = None
            for rs, rid in cands:
                m = xr.match(d, config.WORK_DIR / rs / rid)
                if m and (best is None or m[1] > best[2]):
                    best = (rs, rid, m[1], m[0])
            if best is None:
                say(f"{series}/{sid} #{numbers.get((series, sid))}: no matching edition "
                               f"({len(cands)} tried)")
                continue
            rs, rid, n, H = best
            t = xr.composed_transform(d, config.WORK_DIR / rs / rid, H)
            if t is None:
                say(f"{series}/{sid}: match with {rs}/{rid} left too little overlap")
                continue
            saved = xr.backup(d)
            (d / "rough.json").write_text(json.dumps({
                "transform": runner._transform_json(t, "crossref"), "scale_hint_ft_per_px": None,
                "crossref": {"series": rs, "sheet": rid, "inliers": n}}))
            # refine + score only; the sheet is warped only if the result is kept
            new = runner.run_sheet(by_key[(series, sid)], ref, until="score")
            was = "unplaced" if old is None or not math.isfinite(old.holdout_p50_ft) else \
                f"{old.reason or 'pass'} {old.holdout_p50_ft:.1f} ft"
            now = f"{'pass' if new.passed else new.reason} {new.holdout_p50_ft:.1f} ft"
            if xr.better(old, new):
                shutil.rmtree(saved, ignore_errors=True)
                runner.run_sheet(by_key[(series, sid)], ref)   # warp (the rest is fresh)
                kept += 1
                say(f"{series}/{sid}: KEPT via {rs}/{rid} ({n} matches): {was} -> {now}")
            else:
                xr.restore(d, saved)
                say(f"{series}/{sid}: not better via {rs}/{rid} ({n} matches): "
                               f"{was} vs {now}")
        if kept and publish and not sheet:
            _publish(series)
            paper(year=series, all_=False, min_zoom=11, max_zoom=18, force=False)
    kept_n = sum(" KEPT " in x for x in summary)
    typer.echo(f"done: {kept_n} kept of {len(summary)} flagged sheets tried" if summary else "nothing flagged")


@app.command("eval")
def eval_(
    year: str = typer.Option(..., "--year", help="Series with an answer key: 1948 or 1960."),
) -> None:
    """Per-sheet score next to the distance from our transform to the owner's .points."""
    from sfzmaps import evaluate, pages

    if year not in ("1948", "1960"):
        raise typer.BadParameter("--year must be 1948 or 1960 (the series with answer keys)")
    rows = evaluate.evaluate_series(year, pages.discover())
    if not rows:
        typer.echo(f"no processed sheets for {year}; run `sfzmaps run --year {year}` first")
        raise typer.Exit(1)
    typer.echo(evaluate.format_eval_table(rows))


@app.command()
def bundle() -> None:
    """Bundle processed files into DATA_DIR/share/ for manual copying into Drive (no rclone needed)."""
    from sfzmaps import config as cfg, share

    dest = cfg.DATA_DIR / "share"
    zips = share.bundle(dest)
    typer.echo(f"bundle folder: {dest.resolve()}")
    for f in sorted([*zips, dest / "PROCESSED.json"]):
        typer.echo(f"  {f.name}  {f.stat().st_size:,} bytes")
    typer.echo("Copy these files into the Drive folder old-planning-maps/processed/")


@app.command()
def push() -> None:
    """Optional: bundle and upload to the Drive processed/ folder with rclone (needs rclone set up;
    otherwise use `sfzmaps bundle` and copy the files by hand)."""
    from sfzmaps import share

    share.push()
    typer.echo("pushed")


@app.command()
def pull() -> None:
    """Download processed zips from the public Drive processed/ folder and extract them into data/."""
    from sfzmaps import share

    share.pull()
    typer.echo("pulled")


@app.command()
def edit(
    year: str = typer.Option(None, "--year", help="Only this series in the review queue."),
    sheet: str = typer.Option(None, "--sheet", help="Only this sheet id."),
    port: int = typer.Option(8765, "--port", help="Port on 127.0.0.1."),
    demo: bool = typer.Option(False, "--demo", help="Edit a generated, deliberately corrupted synthetic "
                                                    "sheet in a temporary work dir (no data needed)."),
) -> None:
    """Start the control-point editor (127.0.0.1 only)."""
    import tempfile

    from sfzmaps.editor import server

    if demo:
        from sfzmaps.editor import demo as demo_mod

        root = Path(tempfile.mkdtemp(prefix="sfzmaps-demo-"))
        d = demo_mod.build_demo(root)
        server.set_reference(d.ref)
        server.register_page(d.page)
        year, sheet = d.series, d.sheet_id
        typer.echo(f"demo sheet {d.series}/{d.sheet_id} in {root}")
    server.configure(year=year, sheet=sheet)
    typer.echo(f"editor at http://{server.HOST}:{port}/  (queue: /api/queue)")
    server.serve(port)


@app.command()
def review(
    year: str = typer.Option(..., "--year", help="Series, e.g. 1948."),
    port: int = typer.Option(8000, "--port"),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open the browser."),
) -> None:
    """Serve OUT_DIR on 127.0.0.1 (with Range support) and open the series review page."""
    import webbrowser

    from sfzmaps import reviewpage

    try:
        if reviewpage.is_stale(year):
            typer.echo(f"wrote {reviewpage.write(year)}")
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    srv = reviewpage.make_server(config.OUT_DIR, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/{year}/review.html"
    typer.echo(f"serving {config.OUT_DIR} at {url} (Ctrl-C to stop)")
    if not no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
