import json
import math
from pathlib import Path

import cv2
import typer

from sfzmaps import config, inputs
from sfzmaps.build import WARP_WORKERS
from sfzmaps.sources import drive

app = typer.Typer(no_args_is_help=True, help="Georeference historic SF zoning maps.")
solver_app = typer.Typer(no_args_is_help=True,
                         help="The 2000 solver: make its inputs (each a `sfzmaps build` stage) and solve.")
app.add_typer(solver_app, name="solver")
# Exit status of a command stopped by llm.FATAL (quota, spend cap, missing key): `sfzmaps build`
# reports it as "needs a paid read". Not 2, which click uses for usage errors.
PAID_READ_EXIT = 75
# `sfzmaps build --system-tools` sets this for its stages, so a stage's command (`sfzmaps warp`) accepts the same
# tools. Same value as build.SYSTEM_TOOLS_ENV (a test keeps them equal).
SYSTEM_TOOLS_ENV = "SFZMAPS_SYSTEM_TOOLS"


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
    # the park outlines behind inputs/parks.csv (`sfzmaps parks`)
    if force or not (config.REF_DIR / "rec_park_properties.parquet").exists():
        datasf.fetch_rec_park_properties(config.REF_DIR)
    if force or not (config.REF_DIR / "nps_tracts.parquet").exists():
        datasf.fetch_nps_tracts(config.REF_DIR)
    ref = datasf.load_reference()
    typer.echo(f"blocks: {len(ref.blocks)}")
    typer.echo(f"corners: {len(ref.corners)}")
    typer.echo(f"intersections: {len(ref.intersections)}")
    typer.echo(f"bounds: {list(ref.blocks.total_bounds)}")


@app.command()
def parks(
    out: Path = typer.Option(None, "--out", help="Write the CSV here instead of stdout."),
    maps: bool = typer.Option(False, "--maps", help="Write inputs/park_maps.csv instead: the first "
                              "published map showing each park lot as public."),
) -> None:
    """Park lots the reference data shows (Rec and Park properties, NPS tracts): the candidates
    for inputs/parks.csv, with the NPS conveyance year where there is one. Reads data/ref only.
    --maps: the first map showing each as public (inputs/park_maps.csv), from the published maps."""
    from sfzmaps import parks as park_lots

    if maps:
        path = park_lots.write_first_maps()
        typer.echo(f"{path}")
        return
    text = park_lots.candidates().to_csv(index=False)
    if out:
        out.write_text(text)
    else:
        typer.echo(text, nl=False)


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
    typer.echo(f"saved label scale {best} for {year} to {config._scale_file()} (commit it)")


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
            raise typer.Exit(PAID_READ_EXIT)
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
    """Stitch -> tiles -> manifest for one series (scores read from each sheet's score.json;
    an unchanged layer is not re-tiled: stitch.publish_layer)."""
    from sfzmaps import manifest, reviewpage, stitch, tiles

    scores = stitch.load_scores(series)
    out, report, rebuilt = stitch.publish_layer(series, scores, min_zoom=min_zoom, max_zoom=max_zoom)
    path = manifest.write(series, scores, report)
    typer.echo(f"{series}: {out} ({report.bytes_total / 1e6:.1f} MB of tiles"
               f"{'' if rebuilt else ', unchanged: not re-tiled'}); manifest {path}")
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
    from sfzmaps import fresh, reading, runner, stitch, tiles as tiles_mod, warp
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
            if fresh.is_fresh(out, srcs, force):  # by content, never mtime
                continue
            runner._warp_checked(png, SheetTransform.from_json(json.loads(tj.read_text())),
                                 reading.load_page(pj).mask_boxes, out, paper=True)
            fresh.record(out, srcs)
            n += 1
        try:
            out, report, rebuilt = stitch.publish_layer(series, paper=True, min_zoom=min_zoom,
                                                        max_zoom=max_zoom, force=force)
        except RuntimeError as e:
            typer.echo(f"{series}: skipped ({e})", err=True)
            continue
        typer.echo(f"{series}: {n} sheets warped; {out} ({report.bytes_total / 1e6:.1f} MB"
                   f"{'' if rebuilt else ', unchanged: not re-tiled'})")


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
    shared_archive: bool = typer.Option(False, "--shared", help="Merge every published series into "
                                                                "out/shared/ for the explorer (no "
                                                                "vision calls). Rerun after publishing."),
    force: bool = typer.Option(False, "--force", help="Read and publish even when the outputs are "
                                                     "current (their inputs haven't changed)."),
    check: bool = typer.Option(False, "--check", help="Print the $0 accuracy check of the saved "
                                                     "reads: review cases, answer keys, same-base and "
                                                     "adjacent-era agreement."),
    against: list[str] = typer.Option(None, "--against", help="With --check or --map: also these "
                                                             "dirs of <series>_raw.parquet (a "
                                                             "--snapshot or an experiment)."),
    snapshot: Path = typer.Option(None, "--snapshot", help="Copy every series' current raw reads "
                                                          "to this dir (the 'before' of a change)."),
    map_series: str = typer.Option(None, "--map", help="Draw this series' city map, one panel per "
                                                       "--against dir and one for the current "
                                                       "reads, to data/check/<series>.png."),
) -> None:
    """Read zoning districts per parcel, join the rules, publish out/<series>/districts/. A series
    whose read and outputs are current (nothing they're made from changed) is skipped."""
    import json
    import logging

    import pandas as pd

    from sfzmaps import llm, reviewpage
    from sfzmaps.districts import assemble, shared
    from sfzmaps.districts.series import SERIES

    if check or snapshot or map_series:
        if series or all_ or validate or status or republish or cache_only or sheet or force \
                or shared_archive:
            raise typer.BadParameter("--check, --snapshot and --map run alone")
        from sfzmaps.districts import check as chk

        if snapshot:
            typer.echo(f"{len(chk.snapshot(snapshot))} series copied to {snapshot}")
        panels = {Path(a).name: a for a in against or []}
        if map_series:
            out = chk.city_map(map_series, {**panels, "now": chk.WORK},
                               config.DATA_DIR / "check" / f"{map_series}.png")
            typer.echo(str(out))
        if check:
            typer.echo(chk.report({"now": chk.WORK, **panels}))
        return
    if shared_archive:
        if series or all_ or validate or status or republish or cache_only or sheet or force:
            raise typer.BadParameter("--shared runs alone")
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        names = [s for s, i in SERIES.items() if i.role == "use"] + ["today"]
        typer.echo(f"{shared.build(names)}")
        return
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
                d = assemble.republish(s, force=force)
            else:
                d = assemble.run_series(s, sheets=sheet or None, cache_only=cache_only, force=force)
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
            raise typer.Exit(PAID_READ_EXIT)
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
                    and not inputs.review_path(series, sid).exists():
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
            # The candidate is tried as the sheet's seed (the runner uses a seed over any
            # rough.json), and stays pinned if kept; otherwise the old seed comes back.
            seed_p = inputs.seed_path(series, sid)
            old_seed = seed_p.read_bytes() if seed_p.exists() else None
            runner._write_json(seed_p, {
                "transform": runner._transform_json(t, "crossref"), "scale_hint_ft_per_px": None,
                "crossref": {"series": rs, "sheet": rid, "inliers": n}})
            # refine + score only; the sheet is warped only if the result is kept
            new = runner.run_sheet(by_key[(series, sid)], ref, until="score")
            was = "unplaced" if old is None or not math.isfinite(old.holdout_p50_ft) else \
                f"{old.reason or 'pass'} {old.holdout_p50_ft:.1f} ft"
            now = f"{'pass' if new.passed else new.reason} {new.holdout_p50_ft:.1f} ft"
            if xr.better(old, new):
                shutil.rmtree(saved, ignore_errors=True)
                runner.run_sheet(by_key[(series, sid)], ref)   # warp (the rest is fresh)
                kept += 1
                say(f"{series}/{sid}: KEPT via {rs}/{rid} ({n} matches): {was} -> {now}; pinned as "
                    f"{seed_p.relative_to(config.PIPELINE_DIR)} (commit it)")
            else:
                xr.restore(d, saved)
                if old_seed is None:
                    seed_p.unlink(missing_ok=True)
                else:
                    seed_p.write_bytes(old_seed)
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


@app.command("corrections")
def corrections_cmd(
    issue: int = typer.Argument(..., help="The GitHub issue number the site's corrections tray opened."),
    apply: bool = typer.Option(False, "--apply", help="Write the corrections into inputs/ (overrides "
                               "and review cases). Without it, only check them and write scan crops."),
    file: Path = typer.Option(None, "--file", help="Read the corrections from this file (the tray's "
                              "download, or text pasted into the issue as an attachment) instead of the issue body."),
) -> None:
    """Import readers' corrections from a GitHub issue (districts/corrections.py). First run
    without --apply: check the crops in data/check/corrections/<issue>/ against the proposed codes;
    then --apply, build, and commit inputs/."""
    from sfzmaps.districts import check as chk
    from sfzmaps.districts import corrections as corr
    from sfzmaps.districts.read import load_parcels

    meta = ({"url": f"issue #{issue}", "body": file.read_text()} if file is not None else corr.fetch(issue))
    rows = corr.parse(meta["body"])
    if not rows:
        raise typer.BadParameter(f"issue #{issue} has no ```{corr.FENCE} block")
    parcels = load_parcels(every=True)    # a lot on an old map may since have been retired
    got = corr.check(rows, parcels)
    for p in got.problems:
        typer.echo(f"skipped: {p}", err=True)
    rows = corr.with_twins(got.ok)
    for r in rows:
        typer.echo(f"{r['series']:22} {r['blklot']:10} {r['shown'] or 'none':>8} -> {r['proposed']:8} {r['note']}")
    dest = config.DATA_DIR / "check" / "corrections" / str(issue)
    shots = corr.crops(rows, dest, parcels)
    typer.echo(f"{len(rows)} corrections from {meta['url']}; {len(shots)} scan crops in {dest}")
    if not apply:
        typer.echo("check the crops, then run again with --apply")
        return
    done = corr.write_overrides(rows, issue)
    n_cases = corr.write_cases(rows, issue, chk.centroids())
    typer.echo(f"overrides: {', '.join(f'{s} {n}' for s, n in done.items())}; review cases added: {n_cases}. "
               "Now build and commit inputs/.")


@app.command("answers")
def answers_cmd(
    check: bool = typer.Option(False, "--check", help="List live artifacts whose answers aren't "
                                                     "recorded (no sidecar, no baseline entry)."),
    baseline: bool = typer.Option(False, "--baseline", help="Record every artifact still without a "
                                                           "sidecar in pipeline/cache-baseline.json."),
) -> None:
    """Answer provenance (sfzmaps.answers): which artifacts know the answers they were made from.
    A build replays old artifacts from the cache at $0; --baseline records what's left."""
    from sfzmaps import answers, ship as ship_mod

    if baseline:
        n = answers.write_baseline(answers.all_artifacts())
        typer.echo(f"{answers.baseline_path()}: {n} artifact(s) accepted as they are (commit it)")
        return
    _, unknown = ship_mod.needed_answers()
    for p in unknown:
        typer.echo(str(p))
    typer.echo(f"{len(unknown)} live artifact(s) without recorded answers")
    if check and unknown:
        raise typer.Exit(1)


@app.command("pull-drive")
def pull_drive() -> None:
    """Older way to get data: the processed zips from the public Drive processed/ folder,
    extracted into data/. `sfzmaps pull` gets exactly this commit's cache instead."""
    from sfzmaps import share

    share.pull()
    typer.echo("pulled")


@app.command()
def pull(no_checkout: bool = typer.Option(False, "--no-checkout",
                                          help="Fetch into the caches but don't check data out.")) -> None:
    """This commit's cache from the Blob store (public, no token): the answer packs it records
    and the data its dvc.lock names. Then `sfzmaps build` finds nothing to do, at $0."""
    from sfzmaps import ship as ship_mod

    try:
        packs, objs = ship_mod.pull(dvc_checkout=not no_checkout)
    except ship_mod.ShipError as e:
        typer.echo(f"pull: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"pulled: {packs} answer pack(s) applied, {objs} data object(s) fetched")


@app.command()
def ship(
    base: str = typer.Option("origin/main", "--base", help="What the branch merges into."),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Say what would upload; upload nothing."),
    yes: bool = typer.Option(False, "--yes", help="Ship even if it takes more than 500 requests."),
    repack: bool = typer.Option(False, "--repack", help="Also move small objects stored alone into "
                                "data packs (once, from the old one-object-per-request layout)."),
) -> None:
    """When a branch is ready to merge: upload the answers and data this commit needs that
    `--base` hasn't shipped, to the Blob store, and commit the new answer pack's record (push it
    with the branch). Small data objects travel in data packs, large ones alone. Needs
    site/.env.local (BLOB_READ_WRITE_TOKEN). Never calls the vision API."""
    from sfzmaps import ship as ship_mod

    try:
        r = ship_mod.ship(base, dry_run=dry_run, yes=yes, repack=repack)
    except ship_mod.ShipError as e:
        typer.echo(f"ship: {e}", err=True)
        raise typer.Exit(1)
    verb = "would upload" if dry_run else "uploaded"
    typer.echo(f"{verb} {r.objects_uploaded} of {r.objects_total} data object(s) "
               f"({r.bytes_uploaded / 1e9:.2f} GB) in {r.packs_new} new data pack(s) and "
               f"{r.singles_uploaded} single object(s), and {r.new_answers} new answer(s)"
               + (f" as pack {r.pack[:12]} (record committed; push the branch)" if r.pack else ""))
    typer.echo(f"{'would make' if dry_run else 'made'} about {r.requests} request(s) to the store")


@app.command()
def release(
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Check and plan; publish, commit and push nothing."),
    no_pr: bool = typer.Option(False, "--no-pr", help="Commit the manifests on a release branch; don't push or open a PR."),
) -> None:
    """After a merge: check, export the GIS downloads, publish tiles and downloads to the Blob
    store, and open a PR with the updated manifests (merging it deploys). Needs site/.env.local
    (BLOB_READ_WRITE_TOKEN), or SFZMAPS_ENV_FILE, and `gh`. Never calls the vision API."""
    from sfzmaps import release as release_mod
    from sfzmaps import ship as ship_mod

    try:
        release_mod.release(dry_run=dry_run, no_pr=no_pr, d=release_mod.Deps(echo=typer.echo))
    except (release_mod.ReleaseError, ship_mod.ShipError) as e:
        typer.echo(f"release: {e}", err=True)
        raise typer.Exit(1)


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
    from sfzmaps.districts import assemble

    try:
        assemble.review_tiles(year)     # the districts layer: made here, on demand, not by builds
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


@app.command()
def warp(
    series: list[str] = typer.Argument(None, help="Only these series (default: every series with placed sheets)."),
    jobs: int = typer.Option(WARP_WORKERS, "--jobs", "-j", help="Sheets fitted at once."),
    force: bool = typer.Option(False, "--force", help="Correct sheets again even when their inputs are unchanged."),
    system_tools: bool = typer.Option(False, "--system-tools", envvar=SYSTEM_TOOLS_ENV,
                                      help="Warp with tools that aren't from the pixi environment (not recommended)."),
) -> None:
    """Correct every placed sheet's placement and warp it, into data/warp/<series>/<sheet>/ (skips a sheet whose
    inputs haven't changed). Run inside `pixi shell` so gdalwarp is the pinned one. On the Mac's GPU the fits take a
    shared turn (`sfzmaps.gpu`): they wait while a text detector has the GPU."""
    import logging

    from sfzmaps import build as build_mod
    from sfzmaps import gpu
    from sfzmaps.districts.series import SERIES
    from sfzmaps.sheetwarp import run as sheetwarp
    from sfzmaps.sheetwarp.device import fit_device
    from sfzmaps.sheetwarp.sheet import placed_sheets

    unknown = [s for s in series or [] if s not in SERIES]
    if unknown:
        raise typer.BadParameter(f"unknown series {', '.join(unknown)}; known: {', '.join(SERIES)}")
    build_mod.require_toolchain(system_tools)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failed = []
    with gpu.gpu_turn(False, fit_device()):
        for s in series or [s for s in SERIES if placed_sheets(s)]:
            try:
                sheetwarp.correct_series(s, jobs, force)
            except RuntimeError as e:
                typer.echo(f"error: {e}", err=True)
                failed.append(s)
    if failed:
        raise typer.Exit(1)


@app.command()
def stage(name: str = typer.Argument(..., help="A build stage, e.g. edition-1948, period-1980-1990, "
                                               "today, shared, warp-1948 (see `sfzmaps build --list`).")) -> None:
    """Run one build stage's commands, at $0 (what `sfzmaps build` and `dvc repro` run)."""
    from sfzmaps import build as build_mod

    if name not in build_mod.stages():
        raise typer.BadParameter(f"unknown stage {name!r}; stages: {', '.join(build_mod.stages())}")
    try:
        build_mod.run_stage(name)
    except build_mod.StageFailed as e:
        typer.echo(f"stage {name}: {e}", err=True)
        raise typer.Exit(PAID_READ_EXIT if e.paid_read else 1)


@app.command()
def build(
    targets: list[str] = typer.Argument(None, help="Only these stages (and what they depend on)."),
    jobs: int = typer.Option(4, "--jobs", "-j", help="Stages run at once."),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Say what would run; run nothing."),
    force: bool = typer.Option(False, "--force", help="Run the named stages even if up to date."),
    system_tools: bool = typer.Option(False, "--system-tools", help="Build with tools that aren't "
                                                                    "from the pixi environment (not recommended)."),
    write_dvc_yaml: bool = typer.Option(False, "--write-dvc-yaml", help="Regenerate pipeline/dvc.yaml."),
    list_: bool = typer.Option(False, "--list", help="List the stages."),
) -> None:
    """Rebuild what is stale (by DVC's content hashes), in parallel, never spending money."""
    import logging

    from sfzmaps import build as build_mod

    if write_dvc_yaml:
        build_mod.dvc_yaml_path().write_text(build_mod.dvc_yaml())
        typer.echo(f"wrote {build_mod.dvc_yaml_path()}")
        return
    if list_:
        for st in build_mod.stages().values():
            typer.echo(f"{st.name:24} {' '.join(c[0] + ' ' + c[-1] for c in build_mod.stage_commands(st.name))}")
        return
    if force and not targets:
        raise typer.BadParameter("--force needs the stages to run")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise typer.Exit(build_mod.build(targets or None, jobs, dry_run, force, system_tools))


@app.command()
def downloads(
    series: list[str] = typer.Option(None, "--series", "-s", help="Only this series (repeatable)."),
    system_tools: bool = typer.Option(False, "--system-tools", help="Use GDAL from outside the pixi "
                                                                    "environment (not recommended)."),
) -> None:
    """Write the GIS downloads (lot and district zips, citywide COG, sheet list) into
    data/downloads/ and its index.json; what is unchanged is left alone. $0, no API calls."""
    from sfzmaps import build as build_mod
    from sfzmaps import downloads as dl

    build_mod.require_toolchain(system_tools)
    try:
        index = dl.run(series or None, log=typer.echo)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--series")
    files = index["files"]
    typer.echo(f"{len(files)} files, {sum(f['bytes'] for f in files.values()):,} bytes; "
               f"{sum(len(v) for v in index['sheets'].values())} sheets listed; index {dl.DOWNLOADS_DIR / 'index.json'}")


@app.command()
def doctor(
    renders: int = typer.Option(0, "--renders", help="Also re-render this many PDF pages per series "
                                                     "(into a temp folder) and compare them with "
                                                     "their recorded page_sha256."),
) -> None:
    """The system tools in use, where they come from, and (with --renders) whether they render
    pages byte for byte as the recorded ones (a different render means paid re-reads)."""
    from sfzmaps import toolchain

    ts = toolchain.tools()
    typer.echo(f"toolchain: {toolchain.toolchain_id()}")
    for t in ts:
        where = "pixi" if t.from_env else ("MISSING" if not t.path else "system")
        typer.echo(f"  {t.name:15} {t.version or '':10} {where:7} {t.path or ''}")
    for k, v in toolchain.python_libraries().items():
        typer.echo(f"  {k:26} {v}  (bundled in the Python wheel)")
    bad = toolchain.not_from_env(ts)
    if bad:
        typer.echo(f"{len(bad)} tool(s) not from the pixi environment: run `pixi run doctor`, or "
                   f"this inside `pixi shell` (see pipeline/README.md, Toolchain)")
    if renders:
        n, differ = toolchain.check_renders(renders)
        typer.echo(f"renders: {n} pages checked, {len(differ)} differ"
                   + (f": {', '.join(differ[:10])}" if differ else ""))
        if differ:
            typer.echo("these pages would be re-read (and billed) by labels/classify; don't switch "
                       "toolchains until that's understood", err=True)
            raise typer.Exit(1)
    if bad:
        raise typer.Exit(1)


@solver_app.command("fetch-table")
def solver_fetch_table() -> None:
    """Download the April 2000 zoning table from DataSF into inputs/zoning-2000/, with its PROVENANCE.md."""
    from sfzmaps.solver import table
    info = table.fetch_table()
    typer.echo(f"{info['rows']} rows, sha256 {info['sha256']}")


@solver_app.command("page")
def solver_page() -> None:
    """Merge the warped 2000 sheets into one page on the solver's grid: data/solver/page/ink.png, sheets.png and
    frames.json."""
    import logging

    from sfzmaps.solver import page
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = page.build_page()
    typer.echo(f"{s['sheets']} sheets laid: {s['ink_px']:,} ink px, {s['covered_px']:,} px covered, {s['seconds']} s")


@solver_app.command("lots")
def solver_lots() -> None:
    """Paint the April 2000 lots on the solver's grid: data/solver/lots/lots.npz and table.parquet."""
    from sfzmaps.solver import lots
    s = lots.build_lots()
    typer.echo(f"{s['lots']:,} lots, {s['scored']:,} scored, {s['pairs']:,} neighbour pairs, median centroid of the "
               f"scored lots ({s['median_x']:.3f}, {s['median_y']:.3f}), {s['seconds']} s")


@solver_app.command("prints")
def solver_prints() -> None:
    """Find every printed zoning code on the 2000 sheets with its saved reads (no model is asked):
    data/solver/prints/prints.parquet, terra_boxes.json and summary.json."""
    from sfzmaps.solver import prints
    s = prints.build_prints()
    kinds = s["by_kind"]
    typer.echo(f"{s['prints']:,} prints: Terra and Sol {kinds['0']}, Terra only {kinds['1']}, Sol only {kinds['2']}, "
               f"turned pass only {kinds['3']}; {s['seconds']} s")
    for read, c in s["saved_answers"].items():
        typer.echo(f"  {read}: {c['found']:,} of {c['asked']:,} keys found ({c['missing']:,} missing); "
                   f"{c['unused']:,} of {c['saved']:,} saved answers asked by no print")


@solver_app.command("labels")
def solver_labels() -> None:
    """Fit what the solver knows about the printed labels (no model is asked): data/solver/labels/readings_<q>.npz,
    trust.json, streetside_<q>.npz, lotsize.npz and strokes.npz."""
    from sfzmaps.solver import labels_stage
    s = labels_stage.build_labels()
    t, side, size, ink = s["trust"], s["streetside"], s["lotsize"], s["strokes"]
    typer.echo(f"trust: {t['readings']:,} readings of {t['prints']:,} prints, {t['names']} names; held-out log loss "
               + ", ".join(f"{q} {v['held_out_log_loss']:.4f}" for q, v in t["quarters"].items()))
    typer.echo(f"street sides: {side['on_street']:,} prints on street ground ({side['by_freeway']} by a freeway), "
               f"{side['with_side_b']:,} with a lot across; side A / B right on ordinary streets: "
               + ", ".join(f"{q} {r['rA_st']:.3f} / {r['rB_st']:.3f}" for q, r in side["rates"].items()))
    typer.echo(f"lot size: {size['codes']} codes; codes with no training lots: "
               + ", ".join(f"{q} {n}" for q, n in size["codes_without_lots"].items()))
    typer.echo(f"strokes: {ink['strokes']:,} ({ink['spurs']:,} spurs), {ink['nodes']:,} nodes, {ink['miles']:,} miles")
    typer.echo("seconds: " + ", ".join(f"{k} {v}" for k, v in s["seconds"].items()))


_DEVICE_OPTION = typer.Option(None, "--device", help="cpu or mps (default: mps where available, else cpu).")


def _check_device(device: str | None) -> None:
    if device not in (None, "cpu", "mps"):
        raise typer.BadParameter("cpu or mps", param_hint="--device")


@solver_app.command("text")
def solver_text(device: str = _DEVICE_OPTION) -> None:
    """Train the text detector on every sheet Terra read of the training maps (the validation area included) and
    score every 2000 sheet's page with it: data/solver/text/model.pt and chance/<sheet>.png."""
    import logging

    from sfzmaps.solver.textnet import stage
    _check_device(device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = stage.build_text(device=device)
    typer.echo(f"{s['iterations']:,} steps on {s['device']} over {s['trained_on']} sheets; {s['scored']} 2000 sheets "
               "scored")
    typer.echo("seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in s["seconds"].items()))


def _scores_line(s: dict) -> str:
    """One line of a detector's scores (`scores.score_sheets`' "all", or one map's)."""
    def pc(v, digits=1):
        return "-" if v is None else f"{v:.{digits}%}"

    words = "; ".join(f"{k} {v['found']:,}/{v['of']:,}" for k, v in s["words"].items())
    auc = "-" if s["auc"] is None else f"{s['auc']:.4f}"
    return (f"AUC {auc}, words found {pc(s['words_found'])} ({words}), text ink found {pc(s['text_ink_found'])}, "
            f"ink outside boxes called text {pc(s['ink_outside_boxes_called_text'], 2)}, long-line ink called text "
            f"{pc(s['long_lines']['called_text'], 2)} (of {s['long_lines']['ink_px']:,} px)")


@solver_app.command("text-scores")
def solver_text_scores(device: str = _DEVICE_OPTION) -> None:
    """Score the text detector (data/solver/text/model.pt) on the test drawings, every px of every sheet Terra read
    of them: data/solver/text-scores/scores.json."""
    import logging

    from sfzmaps.solver.textnet import score_stage
    _check_device(device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = score_stage.build_scores(device)
    for name, v in [*s["scores"]["series"].items(), ("all", s["scores"]["all"])]:
        typer.echo(f"{name}: {_scores_line(v)}")
    typer.echo("seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in s["seconds"].items()))


@solver_app.command("text-trial")
def solver_text_trial(
    settings: list[str] = typer.Option(None, "--set", help="One setting of a recipe, name=value (repeatable): "
                                                          "together, one recipe over today's."),
    recipes: list[str] = typer.Option(None, "--recipe",
                                      help="A whole recipe, its settings apart by spaces (\"half_size=True "
                                           "ink_only=ink\"; \"default\" for today's); repeat for several, tried in "
                                           "turn on sheets read once."),
    seed: int = typer.Option(0, "--seed", min=0, help="The seed (its folder gets -seed=N when not 0)."),
    reuse: bool = typer.Option(False, "--reuse", help="Score the model already in each recipe's folder; train "
                                                      "nothing."),
    device: str = _DEVICE_OPTION,
) -> None:
    """Try recipes for the text detector: each trained with the validation area held out on every training map and
    scored there, into data/solver/experiments/text-<recipe>/ (model.pt, recipe.json, scores.json, minutes.json).
    With neither --set nor --recipe, today's recipe."""
    import logging

    from sfzmaps.solver.textnet import trial
    _check_device(device)
    try:
        asked = ([trial.parse_recipe(settings)] if settings else []) + [trial.parse_recipe(r) for r in recipes or []]
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--set / --recipe")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for r in trial.run_trials(asked or [trial.parse_recipe([])], device, seed, reuse):
        typer.echo(f"{r['name']}: {_scores_line(r['scores']['all'])}")
        typer.echo("  minutes: " + ", ".join(f"{k} {v:.1f}" for k, v in r["minutes"].items())
                   + f"; written to {r['folder']}")


def _in_one_thread(argv: list[str]) -> None:
    """Unless this process is pinned to one thread for the maths libraries (`sfzmaps.solver.threads`), run
    `sfzmaps <argv>` again in a process that is, and exit with its status: the libraries read their thread count
    only as they load, and the walls', the placement's and the report's numbers must not depend on how many cores
    the machine has."""
    from sfzmaps.solver import threads
    if not threads.pinned():
        raise typer.Exit(threads.run_pinned(argv))


@solver_app.command("walls")
def solver_walls(
    workers: int = typer.Option(None, "--workers", help="Processes for the closing paths (default half the CPUs)."),
    candidates: str = typer.Option(None, "--candidates",
                                   help="Fit with these candidate features (comma-separated, walls.CANDIDATES; 'none' "
                                        "for none) as an experiment, into data/solver/experiments/walls-<names>/ "
                                        "(for `solve RUN --walls`), never over the stage's walls."),
    evaluate: list[str] = typer.Option(None, "--evaluate",
                                       help="Only score a configuration (comma-separated candidates; 'none' for the "
                                            "default set) and print its held-out log loss; repeat for several, all "
                                            "fitted on inputs read once. Writes nothing."),
) -> None:
    """Fit the wall odds, held out by quarter, from the raw page's strokes: the stage's walls, with
    walls.DEFAULT_CANDIDATES (data/solver/walls/walls_<q>.npz and evaluation.json), or with --candidates an
    experiment's."""
    import logging

    from sfzmaps.solver import walls
    _in_one_thread(["solver", "walls", *(["--workers", str(workers)] if workers is not None else []),
                    *(["--candidates", candidates] if candidates is not None else []),
                    *[a for c in evaluate or [] for a in ("--evaluate", c)]])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def names(text: str) -> list[str]:
        return [c.strip() for c in text.split(",") if c.strip() and c.strip() != "none"]

    if evaluate:
        results = walls.evaluate_candidates([names(c) for c in evaluate], workers)
    else:
        results = [walls.build_walls(workers, None if candidates is None else names(candidates))]
    for s in results:
        typer.echo(f"{', '.join(s['candidates']) or 'no candidates'}: {s['pairs']:,} pairs, {s['features']} "
                   "features; held-out log loss "
                   + ", ".join(f"{q} {v['held_out_log_loss']:.4f} ({v['held_out_pairs']:,} pairs)"
                               for q, v in s["quarters"].items())
                   + (f"; {s['seconds']} s" if "seconds" in s else "")
                   + (f"; written to {s['folder']}" if "folder" in s else ""))


_WALLS_OPTION = typer.Option(None, "--walls", help="The folder of walls_<q>.npz (default data/solver/walls).")


def _not_the_stage_run(run: str) -> None:
    from sfzmaps.solver import runs
    if run == runs.FINAL_RUN:
        raise typer.BadParameter(f"{runs.FINAL_RUN!r} is the solver-solve stage's run, made only by "
                                 f"`sfzmaps solver solve {runs.FINAL_RUN}`", param_hint="RUN")


@solver_app.command("placement")
def solver_placement(
    run: str = typer.Argument(..., help="The experiment's name (data/solver/experiments/RUN/); not 'final'."),
    walls: Path = _WALLS_OPTION,
) -> None:
    """Fit where each printed label belongs, held out by quarter (learning from the hand leader moves of
    label_fixes.csv and the table): data/solver/experiments/RUN/placement.npz."""
    from sfzmaps.solver import placement, runs
    _not_the_stage_run(run)
    try:
        folder = runs.solve_dir(run)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="RUN")
    _in_one_thread(["solver", "placement", run, *(["--walls", str(walls)] if walls else [])])
    placement.fit(folder, walls)


@solver_app.command("solve")
def solver_solve(
    run: str = typer.Argument(..., help="A name for the run: 'final' is the solver-solve stage's (data/solver/solve/), "
                                        "any other an experiment (data/solver/experiments/RUN/)."),
    walls: Path = _WALLS_OPTION,
    true_borders: bool = typer.Option(False, "--true-borders",
                                      help="The ceiling check: the table's own borders as walls (not for 'final')."),
) -> None:
    """Fit the placement, solve the four held-out quarters side by side and join them, list the data errors among
    the run's holes, and score it against J7, into the run's folder made afresh: placement.npz, runs/RUN*.npz (with
    each solve's costs), data_errors.npz, data-errors-2000-digital.csv, report.json and report.md."""
    from sfzmaps.solver import report, runs, solve
    try:
        runs.check_final(run, walls, true_borders)
        runs.solve_dir(run)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="RUN")
    _in_one_thread(["solver", "solve", run, *(["--walls", str(walls)] if walls else []),
                    *(["--true-borders"] if true_borders else [])])
    try:
        found = solve.solve_and_report(run, walls, true_borders)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="RUN")
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(report.markdown(found))


@solver_app.command("solve-quarter")
def solver_solve_quarter(
    run: str = typer.Argument(..., help="The run's name."),
    quarter: str = typer.Argument(..., help="The quarter held out: nw, ne, sw or se."),
    walls: Path = _WALLS_OPTION,
    true_borders: bool = typer.Option(False, "--true-borders"),
    keep_costs: bool = typer.Option(False, "--keep-costs"),
    folder: Path = typer.Option(None, "--folder", help="Where the run is being built (default the run's folder)."),
) -> None:
    """Solve the whole city with one quarter's held-out inputs, with the placement in the run's folder:
    runs/RUN-QUARTER.npz there (no join)."""
    from sfzmaps.solver import paths, solve
    if quarter not in paths.QUARTERS:
        raise typer.BadParameter(f"one of {', '.join(paths.QUARTERS)}", param_hint="QUARTER")
    solve.solve_quarter(run, quarter, walls, true_borders, keep_costs, folder)


@solver_app.command("score")
def solver_score(
    runs: list[str] = typer.Argument(..., help="Runs to score."),
    with_data_errors: bool = typer.Option(False, "--with-data-errors", help="Count the data-error parcels too."),
) -> None:
    """Score runs against the April 2000 table (with its by-eye corrections), each leaving out its own data-error
    list (none for a run without one)."""
    _solver_score(runs, not with_data_errors)


def _solver_score(runs: list[str], exclude_data_errors: bool) -> None:
    from sfzmaps.solver.city import load_run
    from sfzmaps.solver.measures import report
    for r in runs:
        report(r, load_run(r), exclude_data_errors)


@solver_app.command("holes")
def solver_holes(
    run: str = typer.Argument(..., help="A run solved with --keep-costs."),
    picture: Path = typer.Option(None, "--picture", help="Write a picture of every hole, by cause."),
    save_data_errors: bool = typer.Option(False, "--save-data-errors",
                                          help="Write the data-error list into the run's folder (data_errors.npz and "
                                               "data-errors-2000-digital.csv); not for 'final', whose solve writes "
                                               "its own."),
) -> None:
    """Why each hole in a run's district lines is there, and the data errors among them."""
    from sfzmaps.solver.holes import report_causes
    if save_data_errors:
        _not_the_stage_run(run)
    report_causes(run, picture, save_data_errors)


@solver_app.command("published")
def solver_published(
    districts: Path = typer.Option(None, "--districts",
                                   help="districts_raw.parquet (default data/work/2000-digital-zoning/)."),
) -> None:
    """The published 2000 map as the run 'published', scored the same way."""
    from sfzmaps.solver.published import save_published
    _solver_score([save_published(districts)], True)


if __name__ == "__main__":
    app()
