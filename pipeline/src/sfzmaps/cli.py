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
find_app = typer.Typer(no_args_is_help=True, help="The street-and-block finder: one model that finds streets and "
                                                  "block edges on a scanned map.")
app.add_typer(find_app, name="find")
trial_app = typer.Typer(no_args_is_help=True, help="Trying the finder's blocks in placement, in separate data folders "
                                                   "(data/find/trial/<trial>/<arm>/) that never touch today's.")
find_app.add_typer(trial_app, name="trial")
# Exit status of a command stopped by llm.FATAL (quota, spend cap, missing key): `sfzmaps build`
# reports it as "needs a paid read". Not 2, which click uses for usage errors.
PAID_READ_EXIT = 75
# `sfzmaps build --system-tools` sets this for its stages, so a stage's command (`sfzmaps warp`) accepts the same
# tools. Same value as build.SYSTEM_TOOLS_ENV (a test keeps them equal).
SYSTEM_TOOLS_ENV = "SFZMAPS_SYSTEM_TOOLS"


@app.callback()
def main() -> None:
    """Georeference historic SF zoning maps."""


@solver_app.callback()
def solver_main(series: str = typer.Option(None, "--series",
                                           help="The series to solve (default the 2000 map, data/solver/); another "
                                                "series' outputs go to data/solver/<series>/.")
                ) -> None:
    """The solver of one series: the 2000 map unless --series names another."""
    import os
    import sys

    from sfzmaps.solver import paths, threads
    threads.limit_pools()
    if series is None or series == paths.SERIES:
        return
    if not (config.DATA_DIR / "warp" / series).is_dir():
        raise typer.BadParameter(f"no warped sheets of {series} under {config.DATA_DIR / 'warp'}",
                                 param_hint="--series")
    # Every module reads the series once, as it loads (`paths.SERIES`): start the command again in a process whose
    # environment names it, as do the processes it starts.
    cmd = threads.sfzmaps_command() + sys.argv[1:]
    os.execve(cmd[0], cmd, {**os.environ, paths.SERIES_ENV: series})


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


@app.command("frame-labels")
def frame_labels(top: int = typer.Option(30, "--top", help="Stretches per sheet in its crop picture.")) -> None:
    """Rank the ink under every frame label's outline, the map polygons' and the removed blocks' (no GPU).  Writes
    data/frame/labels-qa/<series>/<sheet>.png from the top stretches and prints `series/sheet total-outline-ink
    top-stretch-ink`, highest first."""
    from sfzmaps.frame import data as fd
    from sfzmaps.frame import labels as fl

    rows = []
    for series, sheet in fd.labelled_sheets():
        lum = fd.load_luminance(fd.page_path(series, sheet))
        polys = fd.load_label(series, sheet) + fd.load_remove(series, sheet)
        st = fl.outline_stretches(lum, polys, series, sheet)
        if st:
            fl.qa_sheet(lum, polys, st[:top], fd.OUT / "labels-qa" / series / f"{sheet}.png")
        rows.append((sum(s.ink for s in st), st[0].ink if st else 0.0, series, sheet))
    for total, best, series, sheet in sorted(rows, reverse=True):
        typer.echo(f"{series}/{sheet} {total:.1f} {best:.1f}")


@app.command("frame")
def frame(
    device: str = typer.Option(None, "--device", help="cpu or mps (default: mps where available)."),
    dev: bool = typer.Option(False, "--dev", help="Score the training sheets, each cut by heads that did not train "
                                                  "on it; prints the table and draws the pictures."),
    test: bool = typer.Option(False, "--test", help="Score the test sheets (never trained on), cut by the final "
                                                    "heads; prints the table and draws the pictures."),
    predict_only: bool = typer.Option(False, "--predict-only", help="Decide every page with the saved model, then "
                                                                    "the pictures."),
) -> None:
    """The frame finder (issue #71), $0.  Every mode fits the model first when it is stale (new labels, new pages),
    except `--predict-only`, which refuses.  No flag: decide every map page with the model, draw the pictures.  The
    layout features and heads run on the device in an exclusive GPU turn (`sfzmaps.gpu`); the shape decisions run on
    cpu worker processes, after it.  A sheet labelled both for training and for testing is refused first."""
    import logging

    from sfzmaps import gpu
    from sfzmaps.frame import data as fd
    from sfzmaps.frame import look
    from sfzmaps.frame import run as fr
    from sfzmaps.solver.textnet.train import default_device

    if dev + test + predict_only > 1:
        raise typer.BadParameter("--dev, --test and --predict-only exclude each other")
    device = device or default_device()
    if device not in ("cpu", "mps"):
        raise typer.BadParameter(f"--device must be cpu or mps, not {device!r}")
    try:
        fd.check_split(fd.labelled_sheets(fd.LABELS), fd.test_sheets(fd.TEST_LABELS))
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    out = fd.OUT
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    shown, where = None, out

    def turn():
        return gpu.gpu_turn(True, device)

    try:
        if not (dev or test):
            fr.check_disk(out)
        stale = fr.model_stale(out)
        if stale is not None and predict_only:
            typer.echo(f"error: {stale}: fit the model first (`sfzmaps frame`)", err=True)
            raise typer.Exit(1)
        if stale is not None:
            typer.echo(f"fitting the model first: {stale}")
            fr.fit(device, out, turn=turn)
        if dev:
            shown, where = fr.format_split(fr.score_dev(out)), out / fr.DEV_DIR
        elif test:
            shown, where = fr.format_split(fr.score_test(out)), out / fr.TEST_DIR
        else:
            fr.predict_all(device, out, turn=turn)
    except (fr.DiskFull, fr.StaleFeatures) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    if shown is not None:
        typer.echo(shown)
    look.series_mosaics(where)
    if dev or test:
        typer.echo(f"set mosaic: {look.set_mosaic(out, fr.DEV_DIR if dev else fr.TEST_DIR)}")
    else:
        look.closeups(out)


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
    trust.json, streetside_<q>.npz, lotsize.npz and strokes.npz.  On a series other than 2000, from the 2000 map's
    models (`transfer`)."""
    from sfzmaps.solver import labels_stage, paths
    if not paths.is_2000():
        from sfzmaps.solver import transfer
        typer.echo(json.dumps(transfer.build_labels(), indent=1))
        return
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


@solver_app.command("text-reader")
def solver_text_reader(device: str = _DEVICE_OPTION) -> None:
    """Train the code reader's four text detectors, each with a quarter of the city and the answered blocks held out
    on every training map, and give every 2000 sheet's px the chance of text, the kind and the reading direction of
    the detector that never saw its quarter: data/solver/text-reader/."""
    import logging

    from sfzmaps.solver.textnet import reader_stage
    _check_device(device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = reader_stage.build_reader_text(device=device)
    r = s["radius"]
    typer.echo(f"4 x {s['iterations']:,} steps on {s['device']} over {s['trained_on']} sheets; {s['scored']} 2000 "
               f"sheets scored; {s['blocks']} answered blocks held out")
    typer.echo(f"unknown radius {r['px']} px: {r['longest']['text']} ({r['longest']['series']} "
               f"{r['longest']['sheet']}, {r['longest']['length_px']} px), of {r['words']:,} code words "
               f"({r['left_out']:,} in answered blocks left out)")
    for name, v in s["scores"]["quarters"].items():
        words = "; ".join(f"{k} {w['found']:,}/{w['of']:,}" for k, w in v["words"].items())
        found = "-" if v["words_found"] is None else f"{v['words_found']:.1%}"
        typer.echo(f"{name} on its own quarter: words found {found} ({words})")
    typer.echo("seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in s["seconds"].items()))


@solver_app.command("text-trial")
def solver_text_trial(
    settings: list[str] = typer.Option(None, "--set", help="One setting of a recipe, name=value (repeatable): "
                                                          "together, one recipe over the old recipe (the quarter "
                                                          "models')."),
    recipes: list[str] = typer.Option(None, "--recipe",
                                      help="A whole recipe, its settings apart by spaces (\"half_size=True "
                                           "ink_only=ink\"; \"default\" for the old "
                                           "recipe, \"stage\" first to start from the stage's: \"stage "
                                           "iterations=2000\"); repeat for several, tried in turn on sheets "
                                           "read once."),
    seed: int = typer.Option(0, "--seed", min=0, help="The seed (its folder gets -seed=N when not 0)."),
    reuse: bool = typer.Option(False, "--reuse", help="Score the model already in each recipe's folder; train "
                                                      "nothing."),
    train_series: str = typer.Option(None, "--train-series",
                                     help="Train on these training maps only (comma-separated; default all six); "
                                          "the folder gets -train=<maps>."),
    test_series: str = typer.Option(None, "--test-series",
                                    help="Also score every px of these test drawings (comma-separated): "
                                         "test_scores.json."),
    device: str = _DEVICE_OPTION,
) -> None:
    """Try recipes for the text detector: each trained with the validation area held out on every training map and
    scored there, into data/solver/experiments/text-<recipe>/ (model.pt, recipe.json, scores.json, minutes.json).
    With neither --set nor --recipe, the old recipe (the quarter models'; `stage` as a --recipe's first word starts from
    the stage's)."""
    import logging

    from sfzmaps.solver.textnet import trial
    _check_device(device)

    def names(text: str | None) -> list[str] | None:
        return None if text is None else [s.strip() for s in text.split(",") if s.strip()]

    try:
        maps = trial.training_maps(names(train_series))
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--train-series")
    try:
        tests = trial.test_maps(names(test_series))
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--test-series")
    try:
        asked = ([trial.parse_recipe(settings)] if settings else []) + [trial.parse_recipe(r) for r in recipes or []]
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--set / --recipe")
    asked = asked or [trial.parse_recipe([])]
    folders = [trial.folder(r, seed, maps) for r in asked]
    twice = sorted({f.name for f in folders if folders.count(f) > 1})
    if twice:
        raise typer.BadParameter(f"two of the recipes are one ({', '.join(twice)}; prefetch aside): they would write "
                                 "one folder", param_hint="--set / --recipe")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for r in trial.run_trials(asked, device, seed, reuse, maps, tests):
        typer.echo(f"{r['name']}: {_scores_line(r['scores']['all'])}")
        for name, v in (r["test_scores"] or {"series": {}})["series"].items():
            typer.echo(f"  test {name}: {_scores_line(v)}")
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


@solver_app.command("boxes")
def solver_boxes(workers: int = typer.Option(4, "--workers", help="Sheets grouped at once, one thread each.")) -> None:
    """Word boxes on every 2000 sheet, grouped from the held-out detectors' chance (no outside input): each quarter's
    level chosen on the other three, boxes with their trimmed extent, kind probabilities and map place:
    data/solver/boxes/."""
    import logging

    from sfzmaps.solver import wordboxes
    _in_one_thread(["solver", "boxes", "--workers", str(workers)])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = wordboxes.build_boxes(workers=workers)
    typer.echo(f"{s['boxes']:,} boxes kept; seconds {s['seconds']}")
    for q, v in s["quarters"].items():
        typer.echo(f"{q}: level {v['level']}, {v['words']:,} code words, found {v['recall']:.1%}, "
                   f"axis within 10 degrees {v['axis_within_10']:.1%}")
    typer.echo(f"answered labels: {s['answered']}")


@solver_app.command("text-crops")
def solver_text_crops(workers: int = typer.Option(4, "--workers", help="Sheets cropped at once, one thread each.")
                      ) -> None:
    """The text reader's inputs: crops at Terra's words and non-text windows on the 2000 drawing's three scans, each
    with the quarters and answered blocks it touches; the answered labels' crops; the template reader's font and
    prior; each quarter's synthetic code crops: data/solver/text-crops/."""
    import logging

    from sfzmaps.solver.textread import inputs
    _in_one_thread(["solver", "text-crops", "--workers", str(workers)])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = inputs.build_inputs(workers=workers)
    typer.echo(f"{s['crops']['crops']:,} crops ({s['crops']['by_kind']}); {s['crops']['answered']} answered labels, "
               f"{s['crops']['answered_with_terra_box']} with Terra's box")
    typer.echo(f"template reader: {s['template']['training_labels']} training labels, {s['template']['forms']} forms "
               f"over {s['template']['codes']} codes; training crops per quarter {s['training_crops']}")
    typer.echo("seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in s["seconds"].items()))


@solver_app.command("text-train")
def solver_text_train(device: str = _DEVICE_OPTION) -> None:
    """Train the text reader: for each quarter a CTC reader and a 0/180 classifier that never saw that quarter or an
    answered block (each in its own GPU turn): data/solver/text-train/."""
    import logging

    from sfzmaps.solver.textread import train
    _check_device(device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = train.build_train(device=device)
    for name, h in s["held"].items():
        typer.echo(f"{name}: {h['real_crops']:,} real + {h['synthetic_crops']:,} synthetic crops; monitor CER "
                   f"{h['monitor_cer_last']}")
    typer.echo(f"on {s['device']} in {s['seconds']:.0f} s")


@solver_app.command("text-read")
def solver_text_read(workers: int = typer.Option(4, "--workers", help="Processes reading, one thread each.")) -> None:
    """Read every word box (solver-boxes) with the text reader of its quarter, both ways along its axis, and measure
    on the owner's answers at Terra's boxes and at ours: data/solver/text-read/."""
    import logging

    from sfzmaps.solver.textread import read
    _in_one_thread(["solver", "text-read", "--workers", str(workers)])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = read.build_read(workers=workers)
    t, o = s["terra_boxes"]["all"], s["our_boxes"]["all"]
    typer.echo(f"Terra's boxes, closed top-1: Terra's direction {t['terra_direction'].get('closed_top1')}, axis only "
               f"{t['axis_only'].get('closed_top1')} (n {t['axis_only']['n']})")
    typer.echo(f"our boxes: {o['box_found']} found, {o['box_missed']} missed; closed top-1 on found "
               f"{o['found'].get('closed_top1')}, every label {o['every_label_closed_top1']}")
    typer.echo(f"seconds {s['seconds']}")


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
    chance: str = typer.Option(None, "--chance",
                               help="The text chance the candidates read: 'all-sheets' (the default, the stage's: "
                                    "solver-text's detector) or 'held-out' (solver-text-reader's four held-out "
                                    "detectors). A fit on 'held-out' is an experiment, into "
                                    "data/solver/experiments/walls-<names>-held-out/."),
) -> None:
    """Fit the wall odds, held out by quarter, from the raw page's strokes: the stage's walls, with
    walls.DEFAULT_CANDIDATES (data/solver/walls/walls_<q>.npz and evaluation.json), or with --candidates or
    --chance held-out an experiment's."""
    import logging

    from sfzmaps.solver import textgrid, walls
    textgrid.check_source(chance)
    _in_one_thread(["solver", "walls", *(["--workers", str(workers)] if workers is not None else []),
                    *(["--candidates", candidates] if candidates is not None else []),
                    *[a for c in evaluate or [] for a in ("--evaluate", c)],
                    *(["--chance", chance] if chance is not None else [])])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def names(text: str) -> list[str]:
        return [c.strip() for c in text.split(",") if c.strip() and c.strip() != "none"]

    from sfzmaps.solver import paths
    if not paths.is_2000():                     # another series: the 2000 map's weights on its own features
        from sfzmaps.solver import transfer
        typer.echo(json.dumps(transfer.build_walls(workers or 2), indent=1))
        return
    if evaluate:
        results = walls.evaluate_candidates([names(c) for c in evaluate], workers, chance)
    else:
        results = [walls.build_walls(workers, None if candidates is None else names(candidates), chance)]
    for s in results:
        typer.echo(f"{', '.join(s['candidates']) or 'no candidates'}: {s['pairs']:,} pairs, {s['features']} "
                   "features; held-out log loss "
                   + ", ".join(f"{q} {v['held_out_log_loss']:.4f} ({v['held_out_pairs']:,} pairs)"
                               for q, v in s["quarters"].items())
                   + (f"; {s['seconds']} s" if "seconds" in s else "")
                   + (f"; written to {s['folder']}" if "folder" in s else ""))


_WALLS_OPTION = typer.Option(None, "--walls", help="The folder of walls_<q>.npz (default data/solver/walls).")
_COMPOUND_FUSION_OPTION = typer.Option(
    True, "--compound-fusion/--no-compound-fusion",
    help="The compound fusion move in the search (search.compound_fusion_moves; on by default; 'final' needs it on).")


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
    inputs/hand-labels-2000/label_fixes.csv and the table): data/solver/experiments/RUN/placement.npz."""
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
    compound_fusion: bool = _COMPOUND_FUSION_OPTION,
) -> None:
    """Fit the placement, solve the four held-out quarters side by side and join them, list the data errors among
    the run's holes, and score it against J7, into the run's folder made afresh: placement.npz, runs/RUN*.npz (with
    each solve's costs), data_errors.npz, data-errors-2000-digital.csv, report.json and report.md."""
    from sfzmaps.solver import report, runs, solve
    try:
        solve.check_run(run, walls, true_borders, compound_fusion)
        runs.solve_dir(run)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="RUN")
    _in_one_thread(["solver", "solve", run, *(["--walls", str(walls)] if walls else []),
                    *(["--true-borders"] if true_borders else []),
                    "--compound-fusion" if compound_fusion else "--no-compound-fusion"])
    try:
        found = solve.solve_and_report(run, walls, true_borders, compound_fusion=compound_fusion)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="RUN")
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    from sfzmaps.solver import paths
    typer.echo(report.markdown(found) if paths.is_2000() else json.dumps(found, indent=1))


@solver_app.command("text-chance")
def solver_text_chance(device: str = _DEVICE_OPTION) -> None:
    """Score every sheet of the series with the one text detector (the 2000 map's solver-text model), in one
    exclusive GPU turn: <series' solver folder>/text/chance/<sheet>.png."""
    import logging

    from sfzmaps.solver import transfer
    _check_device(device)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    typer.echo(json.dumps(transfer.build_text_chance(device)))


@solver_app.command("transfer-models")
def solver_transfer_models(
        workers: int = typer.Option(2, "--workers", help="Processes for the walls' closing paths.")) -> None:
    """Fit, on the 2000 map, the models every other series is solved with (walls, trust, street sides, lot sizes,
    placement): data/solver/models-2000/."""
    import logging

    from sfzmaps.solver import transfer
    _in_one_thread(["solver", "transfer-models", "--workers", str(workers)])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    typer.echo(json.dumps(transfer.build_models(workers), indent=1))


@solver_app.command("solve-quarter")
def solver_solve_quarter(
    run: str = typer.Argument(..., help="The run's name."),
    quarter: str = typer.Argument(..., help="The quarter held out: nw, ne, sw or se."),
    walls: Path = _WALLS_OPTION,
    true_borders: bool = typer.Option(False, "--true-borders"),
    keep_costs: bool = typer.Option(False, "--keep-costs"),
    folder: Path = typer.Option(None, "--folder", help="Where the run is being built (default the run's folder)."),
    compound_fusion: bool = _COMPOUND_FUSION_OPTION,
) -> None:
    """Solve the whole city with one quarter's held-out inputs, with the placement in the run's folder:
    runs/RUN-QUARTER.npz there (no join)."""
    from sfzmaps.solver import paths, solve
    if quarter not in paths.QUARTERS:
        raise typer.BadParameter(f"one of {', '.join(paths.QUARTERS)}", param_hint="QUARTER")
    solve.solve_quarter(run, quarter, walls, true_borders, keep_costs, folder, compound_fusion=compound_fusion)


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


@solver_app.command("publish")
def solver_publish(
    series: str = typer.Option("2000-digital-zoning", "--series",
                               help="The zoning series to publish from its solve (1921 ... 2000-digital-zoning)."),
    root: Path = typer.Option(None, "--root", help="The series' solver outputs, in 2000's layout (solve/runs/final.npz, "
                              "solve/report.json, lots/table.parquet, lots/lots.npz); default data/solver for 2000, "
                              "data/solver/<series> for another."),
    page: Path = typer.Option(None, "--page", help="Then write the page of differences against the site's map into "
                              "this folder (index.html, crops/)."),
    published: str = typer.Option("shared", "--published", help="With --page: the map to compare against: 'shared' "
                                  "(the site's archive, data/out/shared) or a parcels.fgb."),
    force: bool = typer.Option(False, "--force", help="Publish even when the outputs are current."),
) -> None:
    """Publish a zoning map from its series' solve (the solver-publish stages): out/<series>/districts/, each lot of
    the read with the solve's code (districts/solved.py), then unzoned blocks, overrides, heights and parks as for
    every series.  Fails if a lot of the read outside the unzoned blocks has no code, or (2000 only) a code has no
    category; another series lists such codes in unknown_codes.json, as the old reader's publish does.
    With --page, then the page of the areas where it differs from the site's map (solver/differences.py)."""
    import logging

    import pandas as pd
    import pyogrio

    from sfzmaps import llm
    from sfzmaps.districts import assemble, solved
    from sfzmaps.districts.series import SERIES
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    llm.CACHE_ONLY = True                 # nothing here may reach the network
    if SERIES.get(series) is None or SERIES[series].role != "use":
        typer.echo(f"{series} is not a zoning series: {', '.join(s for s, i in SERIES.items() if i.role == 'use')}",
                   err=True)
        raise typer.Exit(2)
    if series not in solved.SOLVED:
        typer.echo(f"note: {series} is not in districts.solved.SOLVED, so its edition or period stage still owns "
                   f"data/out/{series}: add it there and `sfzmaps build --write-dvc-yaml` for a build to record this "
                   "publish", err=True)
    try:
        d = solved.publish(series, force=force, root=root)
        with solved.rooted(series, root):
            source = solved.confidence_source(series)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    unknown = json.loads((d / "unknown_codes.json").read_text())
    g = pyogrio.read_dataframe(d / "parcels.fgb", columns=["blklot", "code"], read_geometry=False)
    read = pd.read_parquet(assemble._raw_path(series), columns=["blklot"])["blklot"].astype(str)
    year = SERIES[series].year
    unzoned = assemble.blocks_of(read).map(lambda b: year < assemble.UNZONED_UNTIL.get(b, 0)).to_numpy(bool)
    blank = read[read.isin(set(g.loc[g["code"].isna() | (g["code"] == ""), "blklot"].astype(str))) & ~unzoned]
    typer.echo(f"{series}: {d}; {g['code'].notna().sum():,} of {len(g):,} lots coded; the read's lots "
               f"{len(read):,}, {len(blank):,} of them with no code; codes with no category {len(unknown)}; "
               f"confidence from the solve's {source}")
    if unknown:
        typer.echo(f"no category: {unknown}", err=True)
    # 2000 must publish every code with a category; another series keeps a code its era's research table can't
    # place, with category unknown and listed in unknown_codes.json, as the old reader's publish does
    if (unknown and series == solved.SERIES) or len(blank):
        typer.echo(f"no code: {list(blank[:10])}", err=True)
        raise typer.Exit(1)
    if page is not None:
        from sfzmaps.solver import differences
        out = differences.make(series, root=root, published=published, page_dir=page, new_dir=d)
        typer.echo(f"the page of differences: {out}")


@find_app.command("sheets")
def find_sheets(series: str = typer.Option(None, "--series", help="Only this series (default: every series).")) -> None:
    """Print the map pages the finder reads, one `series/sheet` a line (duplicate scans left out)."""
    from sfzmaps.find import pages
    for s, h in pages.sheets(series):
        typer.echo(f"{s}/{h}")


@find_app.command("labels")
def find_labels(series: str = typer.Option(..., "--series", help="A series whose lots label it (label_sources.csv)."),
                sheet: list[str] = typer.Option(None, "--sheet", help="Only these sheets (repeatable; default: all)."),
                placement: str = typer.Option("warp", "--placement",
                                              help="The sheets' placement: warp (data/warp) or work (data/work)."),
                look: bool = typer.Option(False, "--look", help="Also draw each sheet's labels on its scan "
                                                                "(data/find/look/labels/<placement>/).")
                ) -> None:
    """Make the series' street-or-not labels from its lots on the solver's grid (data/find/labels/<series>/grid.npz,
    made again every run), then each sheet's labels through its placement: street or not, no block edge, and the cuts
    saying where a block edge crosses (data/find/labels/<series>/<sheet>.npz).  Street labels are reported by their px
    and the sum of their weights (most px of a street label far from its centre line weigh almost nothing).  The lots
    listed in inputs/find/unlabelled_lots.csv (the two parks held as one lot) carry no label; a listed lot that is not
    among the series' lots is refused.  $0."""
    import time

    import numpy as np

    from sfzmaps.find import look as pictures, lotlabels, pages
    from sfzmaps.solver.hatchnet.data import NOT_STREET, STREET, UNKNOWN
    if placement not in lotlabels.PLACEMENTS:
        raise typer.BadParameter(f"expected one of {', '.join(lotlabels.PLACEMENTS)}", param_hint="--placement")
    try:
        pages.source(series)
    except KeyError as e:
        raise typer.BadParameter(str(e.args[0]), param_hint="--series") from None
    found = [h for _, h in pages.sheets(series)]
    unknown = sorted(set(sheet or []) - set(found))
    if unknown:
        raise typer.BadParameter(f"not a map page of {series} that is read: {', '.join(unknown)}", param_hint="--sheet")
    try:
        lotlabels.unlabelled_lots()
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"the list of lots carrying no label ({lotlabels.UNLABELLED}) cannot be read: {e}", err=True)
        raise typer.Exit(1) from None
    try:
        g = lotlabels.make_ground(series)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--series") from None
    for name, v in (("street", STREET), ("block (not street)", NOT_STREET), ("no label", UNKNOWN)):
        typer.echo(f"{name}: {int(np.count_nonzero(g.street == v)):,} grid px")
    px, weighs = lotlabels.street_totals(g.street, g.weight)
    typer.echo(f"street labels: {px:,} grid px with weight above 0, weighing {weighs:,.0f} in all")
    typer.echo(f"listed lots carrying no label ({lotlabels.UNLABELLED.name}): {int(np.count_nonzero(g.unlabelled)):,} "
               f"grid px")
    typer.echo(f"the grid labels: {lotlabels.grid_path(series)}")
    for h in sheet or found:
        t = time.time()
        lab = lotlabels.sheet_labels((series, h), placement)
        typer.echo(f"{h}: sigma {lab['sigma']:.3f} px; block-edge cuts {len(lab['cuts'].weight):,}; "
                   f"{time.time() - t:.1f} s; {lotlabels.sheet_path((series, h))}")
        n = lab["counts"]
        kept = n["labelled_px_after"] / max(1, n["labelled_px_before"])
        px, weighs = lotlabels.street_totals(lab["street"], lab["weight"])
        typer.echo(f"  street labels: {px:,} px with weight above 0, weighing {weighs:,.0f} in all; listed lots "
                   f"left without a label: {n['unlabelled_lot_px']:,} px")
        typer.echo(f"  where the sheet draws ink: block-edge cuts used {n['edge_used']:,}, dropped "
                   f"{n['edge_dropped']:,}; labelled px kept {kept:.3f}")
        if look:
            for f in pictures.write((series, h), pages.load_scan((series, h)), lab):
                typer.echo(f"  {f}")


@find_app.command("crops")
def find_crops(set_: str = typer.Option(None, "--set", help="Only this set: train, choose or test (default: all)."),
               look: bool = typer.Option(False, "--look", help="Draw each crop's scan and labels side by side "
                                                               "(data/find/look/crops/<set>/).")) -> None:
    """Make the labels of every hand crop (inputs/find-labels/<set>/<series>/<sheet>__<x0>_<y0>_<w>_<h>.json) and
    print each crop's areas by kind and its warnings.  A crop file that cannot be read or labelled is reported with its
    path and the rest go on.  Then every choosing crop lying on a test crop's ground, sent through the sheets'
    placements, is printed with the share of its area that overlaps.  The exit status is 1 if any crop failed or any
    such overlap was found.  $0."""
    from sfzmaps.find import handlabels, look as pictures, pages
    if set_ is not None and set_ not in handlabels.SETS:
        raise typer.BadParameter(f"expected one of {', '.join(handlabels.SETS)}", param_hint="--set")
    found = handlabels.crop_files(handlabels.SETS if set_ is None else (set_,))
    if not found:
        typer.echo("no crops")
    failed, scans = 0, {}
    for path, s in found:
        name = f"{s}/{path.parent.name}/{path.name}"
        try:
            c = handlabels.read_crop(path, s)
            sid = (c.series, c.sheet)
            if sid not in scans:
                scans = {sid: pages.load_scan(sid)}         # crops come sorted by sheet: keep one page in memory
            lum = handlabels.crop_scan(scans[sid], c)
            lab = handlabels.make_labels(lum, c)
        except (ValueError, FileNotFoundError) as e:
            failed += 1
            msg = str(e)
            typer.echo(f"{name}: FAILED: {msg if str(path) in msg else f'{path}: {msg}'}")
            continue
        kinds = list(lab.kinds.values())
        typer.echo(f"{name}: block {kinds.count('block')}, open {kinds.count('open')}")
        for w in lab.warnings:
            typer.echo(f"  {w}")
        if look:
            typer.echo(f"  {pictures.write_crop(c, lum, lab)}")
    overlaps = []
    try:
        overlaps = handlabels.choose_on_test(handlabels.crops(("choose",)), handlabels.crops(("test",)))
    except (ValueError, FileNotFoundError) as e:
        failed += 1
        typer.echo(f"choosing crops against test crops: not checked: {e}")
    for c, t, share in overlaps:
        typer.echo(f"choose-against-test ground overlap: {c.set}/{c.series}/{c.path.name} lies on the ground of the "
                   f"test crop {t.set}/{t.series}/{t.path.name} ({share:.3f} of the choosing crop's area)")
    if failed:
        typer.echo(f"{failed} of {len(found)} crops failed")
    if overlaps:
        typer.echo(f"{len(overlaps)} choose-against-test overlap(s): a choosing crop must lie away from every test "
                   f"crop's ground")
    if failed or overlaps:
        raise typer.Exit(1)


@find_app.command("train")
def find_train(run: str = typer.Option(..., "--run", help="The run's name: its files go to data/find/runs/<run>/."),
               sheet: list[str] = typer.Option(None, "--sheet", help="A lot-labelled sheet to train on, as "
                                                                     "<series>/<sheet> (repeatable)."),
               unbroken: bool = typer.Option(False, "--unbroken", help="Add the unbroken-line terms on the hand "
                                                                       "crops' block edges."),
               no_noise: bool = typer.Option(False, "--no-noise", help="No print-and-scan noise on the tiles."),
               no_resize: bool = typer.Option(False, "--no-resize", help="Show every tile at its item's own print size "
                                                                         "(for comparison), none at the measured "
                                                                         "print sizes of the sheets."),
               steps: int = typer.Option(None, "--steps", min=1, help="Engineering setting: train this many steps "
                                                                      "instead of the recipe's (a short check)."),
               every: int = typer.Option(None, "--every", min=1, help="Engineering setting: steps between "
                                                                      "checkpoints instead of the recipe's."),
               no_trace: bool = typer.Option(False, "--no-trace", help="Skip scoring the choosing crops during "
                                                                       "training."),
               seed: int = typer.Option(0, "--seed", help="The recipe's seed: the tile draws, the noise, the print "
                                                         "sizes drawn and the network's start all derive from it."),
               device: str = typer.Option(None, "--device", help="cpu or mps (default: mps where there is one)."),
               look: bool = typer.Option(False, "--look", help="Also draw training tiles without and with noise "
                                                               "(data/find/runs/<run>/noise-examples.png).")) -> None:
    """Train the finder on the named lot-labelled sheets (`sfzmaps find labels`) and every `train` hand crop, the
    ground under the choosing and test crops left out: data/find/runs/<run>/model.pt and log.json (the recipe, device,
    commit (null, with one line saying why, when git cannot tell), items, curve, and the print sizes measured and the
    factors drawn).  Each tile is shown at the print size of a sheet the finder is used on, measured from every sheet
    of every series whose placement passes the warp's own check (`find.data.print_sizes`; with --no-resize, at its
    item's own size).  The last step's weights are kept.  At
    every checkpoint the `choose` crops are read and the share of their blocks found is logged in the curve as the
    run's held-out trace (`find.counts.trace_scorer`: the share of their blocks found and the panoptic quality of
    their areas, the connected pieces of ground the model calls not street, per checkpoint; it chooses nothing;
    --no-trace skips it).  --seed sets the recipe's seed (0 by default).  Every training crop (and, unless
    --no-trace, every choosing crop) is read first; if any cannot be, each is reported with its path and nothing is
    trained (exit status 1), as when there is no choosing crop and --no-trace is not given.  $0."""
    import dataclasses
    import logging
    import math
    import time

    import torch

    from sfzmaps import gpu
    from sfzmaps.find import counts, data, handlabels, pages, train
    from sfzmaps.solver import hatching
    from sfzmaps.solver.hatchnet.train import THREADS, default_device
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not run or run == "." or ".." in run or "/" in run or "\\" in run:
        raise typer.BadParameter(f"{run!r}: a run's name is one folder name (no path separator, no '..')",
                                 param_hint="--run")
    recipe = train.Recipe()
    recipe = dataclasses.replace(recipe, steps=steps or recipe.steps, every=every or recipe.every, unbroken=unbroken,
                                 noise=0.0 if no_noise else recipe.noise, resize=not no_resize, seed=seed)
    device = device or default_device()
    sids = []
    for s in sheet or []:
        series, _, name = s.partition("/")
        if not series or not name:
            raise typer.BadParameter(f"{s}: expected <series>/<sheet>", param_hint="--sheet")
        if name in hatching.duplicate_scans(series):
            raise typer.BadParameter(f"{s} is a duplicate scan, which is never read", param_hint="--sheet")
        sids.append((series, name))
    def said(e: Exception) -> str:                     # a KeyError's message is its first argument (str() quotes it)
        return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)

    items, failed, on_sheet = [], 0, []                # on_sheet: each item's (series, sheet)
    try:
        for sid in sids:
            items.append(data.sheet_labelled(sid))
            on_sheet.append(sid)
    except (FileNotFoundError, ValueError, KeyError) as e:
        typer.echo(said(e), err=True)
        raise typer.Exit(1) from None
    for path, s in handlabels.crop_files(("train",)):
        try:
            crop = handlabels.read_crop(path, s)
            items.append(data.crop_labelled(crop))
            on_sheet.append((crop.series, crop.sheet))
        except (ValueError, FileNotFoundError, KeyError) as e:
            failed += 1
            msg = said(e)
            typer.echo(f"FAILED: {msg if str(path) in msg else f'{path}: {msg}'}", err=True)
    if failed:
        typer.echo(f"{failed} training crop(s) could not be read; nothing trained", err=True)
        raise typer.Exit(1)
    choosing, on_checkpoint = [], None
    if not no_trace:
        try:
            choosing = handlabels.crops(("choose",))
            on_checkpoint = counts.trace_scorer(choosing, device)
        except (ValueError, FileNotFoundError, KeyError) as e:
            typer.echo(f"the choosing crops: {said(e)}; they give the training trace (--no-trace trains without "
                       f"it); nothing trained", err=True)
            raise typer.Exit(1) from None
    if not items:
        typer.echo("nothing to train on", err=True)
        raise typer.Exit(1)
    listed = [{"name": it.name, "era": it.era, "ft_per_px": it.ft_per_px, "labelled_px": int(data.labelled(it).sum()),
               "used_cuts_with_no_px": 0 if it.cuts is None else data.cuts_without_px(it.cuts),
               **({} if it.hand else {"hand_marked_px": it.hand_px, "hand_marked_street_weight": it.hand_weight})}
              for it in items]
    for it in items:
        if not it.hand and it.hand_px:
            typer.echo(f"{it.name}: {it.hand_px:,} px left without a lot label as hand-marked ground (street weight "
                       f"{it.hand_weight:,.1f})")
    for era in sorted({it.era for it in items}):
        mine = [x for x in listed if x["era"] == era]
        typer.echo(f"{era}: {len(mine)} items, {sum(x['labelled_px'] for x in mine):,} labelled px")
    sizes, size_fields = None, {}
    if recipe.resize:
        try:
            sizes, left_out, sheets_left_out = data.print_sizes()
        except (FileNotFoundError, ValueError, KeyError) as e:
            typer.echo(f"the print sizes: {said(e)}; nothing trained", err=True)
            raise typer.Exit(1) from None
        size_fields = {"print_sizes": data.size_summary(sizes), "print_sizes_left_out": left_out,
                       "print_sizes_sheets_left_out": {"count": len(sheets_left_out), "sheets": sheets_left_out},
                       "training_items_on_sheets_left_out": [it.name for it, sid in zip(items, on_sheet)
                                                             if f"{sid[0]}/{sid[1]}" in sheets_left_out]}
        typer.echo(f"print sizes: {sum(len(v) for v in sizes.values())} placed sheets of {len(sizes)} series, "
                   f"{min((min(v) for v in sizes.values()), default=math.nan):.2f}-"
                   f"{max((max(v) for v in sizes.values()), default=math.nan):.2f} ft per px; left out (no placed "
                   f"sheet): {', '.join(left_out) or 'none'}; {len(sheets_left_out)} sheet(s) left out: "
                   f"{', '.join(f'{k} ({v})' for k, v in sheets_left_out.items()) or 'none'}")
    out = pages.OUT / "runs" / run
    if look:
        typer.echo(str(train.write_noise_examples(items, recipe, out / "noise-examples.png", sizes=sizes)))
    changed, why = _git("status", "--porcelain", "--untracked-files=no", "--", "src")
    t0 = time.perf_counter()
    try:
        with gpu.gpu_turn(True, device):
            fitted = train.fit(items, recipe, device, on_checkpoint, sizes=sizes)
    except ValueError as e:
        typer.echo(f"nothing trained: {e}", err=True)
        raise typer.Exit(1) from None
    commit, why_head = _git("rev-parse", "HEAD")
    why = why or why_head
    if why:
        typer.echo(f"the git state was not recorded (commit and uncommitted_changes_in_src are null): {why}")
    fields = {"commit": commit, "uncommitted_changes_in_src": None if changed is None else bool(changed),
              "torch": torch.__version__,
              "threads": THREADS, "items": listed, "choosing_crops": [data.crop_name(c) for c in choosing] or None,
              "train_crops_left_out_as_reserved": {it.name: it.left_out for it in items if it.hand}, **size_fields,
              "seconds": round(time.perf_counter() - t0, 1)}
    for f in train.save_run(out, fitted, recipe, fields):
        typer.echo(str(f))


def _git(*args: str) -> tuple[str | None, str | None]:
    """git's output (stripped) for `args`, run in the pipeline folder, and None; or None and why, in one line, when git
    is not installed, fails or finds no repository there (a run goes on without the git state)."""
    import subprocess
    try:
        done = subprocess.run(["git", *args], cwd=config.PIPELINE_DIR, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return None, "git is not installed"
    except subprocess.CalledProcessError as e:
        said = next((line for line in (e.stderr or "").splitlines() if line.strip()), "")
        return None, f"`git {' '.join(args)}` exited {e.returncode}" + (f" ({said.strip()})" if said else "")
    return done.stdout.strip(), None


@find_app.command("count")
def find_count(run: str = typer.Option(..., "--run", help="The run (data/find/runs/<run>/model.pt) to count."),
               device: str = typer.Option(None, "--device", help="cpu or mps (default: mps where there is one).")
               ) -> None:
    """Count what the run's kept model finds on every `choose` and every `test` hand crop, each set measured alike and
    on its own.  The found areas are the connected pieces of ground the model calls not street.  The measures: the
    panoptic quality (how well the found areas match the answer key's areas, counting stray areas and missed ones); how
    many of the answer key's blocks are found, joined, cut up, lost or held whole, how much of the answer key's street
    and block ground it calls so, how far the answer key's block boundary lies from the nearest block edge the model
    drew and how far each block edge the model drew lies from the nearest answer-key boundary, in feet; per crop, per
    era and pooled (`find.counts`).  The answer key: block outlines marked by hand on the scan, never shown to the
    model.  Writes data/find/runs/<run>/counts.json, each crop's read (reads/<set>/<series>__<crop>.npz) and picture
    (look/<set>/<series>__<crop>.png).  Exit status 1 when the model
    predates the two-answer finder, there is no choosing or no test crop, a choosing crop lies on a test crop's
    ground, or a crop cannot be read.  $0."""
    import json
    import logging

    import torch

    from sfzmaps import gpu
    from sfzmaps.find import counts, handlabels, pages
    from sfzmaps.solver.hatchnet.train import default_device, settle_torch
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not run or run == "." or ".." in run or "/" in run or "\\" in run:
        raise typer.BadParameter(f"{run!r}: a run's name is one folder name (no path separator, no '..')",
                                 param_hint="--run")
    folder = pages.OUT / "runs" / run
    if not (folder / "model.pt").is_file():
        typer.echo(f"{folder / 'model.pt'}: no model; train one with `sfzmaps find train --run {run}`", err=True)
        raise typer.Exit(1)
    device = device or default_device()
    saved = torch.load(folder / "model.pt", map_location="cpu")
    settle_torch()
    try:
        net = counts.network(saved["state_dict"])
        choose, test = handlabels.crops(("choose",)), handlabels.crops(("test",))
        with gpu.gpu_turn(True, device):
            out = counts.count(net, choose, test, device, folder)
    except (ValueError, FileNotFoundError, KeyError) as e:
        typer.echo(str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e), err=True)
        raise typer.Exit(1) from None
    out = {"run": run, "step": saved["step"], "device": str(device), **out}
    f = folder / "counts.json"
    f.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    for set_, said in (("choose", "choosing crops"), ("test", "test crops")):
        t = out[set_]
        pooled_q = t["panoptic"]["pooled"]["quality"]
        typer.echo(f"{said}: panoptic quality {t['panoptic']['mean_over_eras']:.3f}, find {t['found_share']:.3f} of "
                   f"their blocks (means over eras); pooled: {t['pooled']}, panoptic quality "
                   f"{'none' if pooled_q is None else f'{pooled_q:.3f}'}, held whole {t['whole']['pooled']['whole']} "
                   f"of {t['whole']['pooled']['blocks']}")
        for era, c in t["eras"].items():
            g, b, h = t["ground"]["eras"][era], t["boundary_feet"]["eras"][era], t["whole"]["eras"][era]
            m = t["model_edge_feet"]["eras"][era]
            back = ("not measured (no model edge px, or no answer-key boundary px)" if m["median"] is None else
                    f"median {m['median']:.1f} ft, 90% {m['p90']:.1f} ft, 99% {m['p99']:.1f} ft")
            feet = "no model edge" if b["median"] is None else (f"median {b['median']:.1f} ft, 90% {b['p90']:.1f} ft, "
                                                                f"99% {b['p99']:.1f} ft")
            pq = t["panoptic"]["eras"][era]["quality"]
            quality = "none" if pq is None else f"{pq:.3f}"
            typer.echo(f"  {set_} {era}: {c}; panoptic quality {quality}; held whole"
                       f" {h['whole']} of {h['blocks']}; street px called street "
                       f"{g['street_share']}, area px called not street {g['area_share']}; answer key's block "
                       f"boundary to the nearest block edge the model drew: {feet}; block edge the model drew to the "
                       f"nearest answer-key boundary: {back}")
    typer.echo(str(f))


@find_app.command("read")
def find_read(run: str = typer.Option(..., "--run", help="The run (data/find/runs/<run>/model.pt) to read with."),
              sheet: list[str] = typer.Option(None, "--sheet", help="A sheet to read, as <series>/<sheet> "
                                                                    "(repeatable)."),
              series: str = typer.Option(None, "--series", help="Read every sheet of this series (duplicate scans "
                                                                "skipped and named)."),
              device: str = typer.Option(None, "--device", help="cpu or mps (default: mps where there is one).")
              ) -> None:
    """Read whole sheets with the run's kept model: the page the placement steps use (data/work/<series>/<sheet>/
    page.png) and its frame cut, at the page's own px.  For each sheet it writes the read (the chance of street and of
    a block edge at every px, stored as levels: data/find/runs/<run>/sheets/<series>/<sheet>.npz) and a RESULT picture
    at the page's own size (the scan with the ground the model calls street tinted orange and its block edges drawn:
    .png beside it), and prints one line: the sheet, its size, the share of the map area called street and the
    seconds it took.  Give --sheet (one or more) or --series; nothing else is read.  Exit status 1, reading nothing,
    when an asked sheet is a duplicate scan (never read), has no page image or no frame cut, or the run has no model;
    with --series, duplicate scans are skipped and named.  $0."""
    import torch

    from sfzmaps import gpu
    from sfzmaps.find import counts, pages, wholesheet
    from sfzmaps.solver.hatchnet.train import default_device, settle_torch
    if not run or run == "." or ".." in run or "/" in run or "\\" in run:
        raise typer.BadParameter(f"{run!r}: a run's name is one folder name (no path separator, no '..')",
                                 param_hint="--run")
    if not sheet and not series:
        raise typer.BadParameter("give the sheets to read (--sheet <series>/<sheet>) or a series (--series)",
                                 param_hint="--sheet / --series")
    sids = []
    for s in sheet or []:
        ser, _, name = s.partition("/")
        if not ser or not name:
            raise typer.BadParameter(f"{s}: expected <series>/<sheet>", param_hint="--sheet")
        sids.append((ser, name))
    if series:
        try:
            found, dups = pages.sheets_and_duplicates(series)
        except FileNotFoundError as e:
            typer.echo(f"series {series}: {e}; nothing read", err=True)
            raise typer.Exit(1) from None
        if not found:
            typer.echo(f"no sheets in series {series} (no map page of it is read); nothing read", err=True)
            raise typer.Exit(1)
        for (s, h), kept in dups:
            typer.echo(f"{s}/{h}: skipped, a duplicate scan of {kept} (never read)")
        sids += [sid for sid in found if sid not in sids]
    folder = pages.OUT / "runs" / run
    refused = [why for why in (wholesheet.check(sid) for sid in sids) if why]
    if not (folder / "model.pt").is_file():
        refused.append(f"{folder / 'model.pt'}: no model; train one with `sfzmaps find train --run {run}`")
    if refused:
        for why in refused:
            typer.echo(why, err=True)
        typer.echo("nothing read", err=True)
        raise typer.Exit(1)
    device = device or default_device()
    saved = torch.load(folder / "model.pt", map_location="cpu")
    settle_torch()
    try:
        net = counts.network(saved["state_dict"])
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    with gpu.gpu_turn(True, device):
        for sid in sids:
            got = wholesheet.read_sheet(net, sid, device, folder, run)
            h, w = got["shape"]
            typer.echo(f"{sid[0]}/{sid[1]}: {h} x {w} px; {got['street_share']:.3f} of the map area called street; "
                       f"{got['seconds']:.1f} s; {got['read']}; {got['picture']}")


@find_app.command("ruled")
def find_ruled(sheet: str = typer.Option(..., "--sheet", help="The page, as <series>/<sheet>."),
               box: str = typer.Option(..., "--box", help="The page px box x0,y0,w,h."),
               zoom: int = typer.Option(4, "--zoom", help="The tiles' enlargement (nearest neighbour)."),
               tiles: int = typer.Option(2, "--tiles", help="Cut the box into this many x this many tiles.")) -> None:
    """Close-ups of a page box to draw a hand crop from: the scan cut into tiles, each enlarged, with page-px rulers
    and a light grid every 50 px, and the whole box at 2x (data/find/look/ruled/<series>__<sheet>__<box>/).  $0."""
    from sfzmaps.find import look as pictures, pages
    from sfzmaps.solver import hatching
    series, _, name = sheet.partition("/")
    if not series or not name:
        raise typer.BadParameter("expected <series>/<sheet>", param_hint="--sheet")
    if name in hatching.duplicate_scans(series):
        raise typer.BadParameter(f"{sheet} is a duplicate scan, which is never read", param_hint="--sheet")
    try:
        x0, y0, w, h = (int(v) for v in box.split(","))
    except ValueError:
        raise typer.BadParameter("expected x0,y0,w,h (whole numbers)", param_hint="--box") from None
    if zoom < 1 or tiles < 1:
        raise typer.BadParameter("--zoom and --tiles must be at least 1")
    lum = pages.load_scan((series, name))
    if w < tiles or h < tiles or x0 < 0 or y0 < 0 or x0 + w > lum.shape[1] or y0 + h > lum.shape[0]:
        raise typer.BadParameter(f"must lie on the page ({lum.shape[1]} x {lum.shape[0]} px) and hold every tile",
                                 param_hint="--box")
    for f in pictures.write_ruled((series, name), lum[y0:y0 + h, x0:x0 + w], (x0, y0, w, h), zoom, tiles):
        typer.echo(str(f))


OFFSET_MEANS = ("offset: how far the finder's area (ground it calls not street) reaches past today's traced block "
                "outline, in px, measured at every px along the outline; positive = past the outline, towards the "
                "street; negative = it stops short.  The area's edge is taken half a px beyond its outer px.  The "
                "tracing step takes an outlined (mostly white) block's contour through the centres of its outer ink "
                "px, half a px inside the line's outer edge, and moves it in by half the line width it measures, so "
                "an area ending at the line's outer edge measures half the measured line width + 0.5, and one ending "
                "on the traced outline 0.  For an odd line width the measured width is the line's own, so the traced "
                "outline sits half a px inside the line's true centre.  A dark (solid or hatched) block is not moved "
                "in: an area ending at its ink's edge measures +0.5.")


@find_app.command("outline-offset")
def find_outline_offset(run: str = typer.Option(..., "--run", help="The run whose stored whole-sheet reads are "
                                                                  "measured (data/find/runs/<run>/sheets/)."),
                        trial: str = typer.Option(..., "--trial", help="The trial's name: the result goes to "
                                                                      "data/find/trial/<trial>/outline_offsets.json."),
                        sheet: list[str] = typer.Option(..., "--sheet", help="A sheet whose tracing works, as "
                                                                             "<series>/<sheet> (repeatable; one per "
                                                                             "era).")) -> None:
    """Measure where the edge of the finder's areas sits against today's traced block outlines, per era: on the
    centre of the printed line, or on its outer edge.  For each sheet it reads the run's stored read of it, its frame
    cut, its page and today's traced blocks (data/work/<series>/<sheet>/blocks_px.geojson, read only), and at every px
    along each traced outline measures how far the finder's area reaches past the outline (positive: past it, towards
    the street).  The answer is measured on the traced blocks that lie in a finder area holding no other traced block
    (an offset on a block the finder joined to its neighbour means little); the figures for every traced block, and
    for mostly white and mostly ink blocks apart (by the tracing step's own rule: more of its seed px inside the block
    paper than ink), are written beside it, with the sheet's half line width (half `linework.stroke_width` of its ink).
    A report only: nothing reads it to decide anything.  Prints one line per era and writes only
    data/find/trial/<trial>/outline_offsets.json.  Exit status 1, writing nothing, when a sheet
    has no stored read (naming the `sfzmaps find read` command), is a duplicate scan, has no page, frame cut, traced
    blocks or era, or shares its era with another sheet given.  $0."""
    import numpy as np
    from shapely.geometry import shape

    from sfzmaps import linework
    from sfzmaps.find import areas, pages, placeblocks, read, trial as trials, wholesheet
    from sfzmaps.frame import data as frame_data
    for name, hint in ((run, "--run"), (trial, "--trial")):
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise typer.BadParameter(f"{name!r}: one folder name (no path separator, no '..')", param_hint=hint)
    sids = []
    for s in sheet:
        ser, _, name = s.partition("/")
        if not ser or not name:
            raise typer.BadParameter(f"{s}: expected <series>/<sheet>", param_hint="--sheet")
        sids.append((ser, name))
    folder = pages.OUT / "runs" / run
    refused, eras = [], {}
    for sid in sids:
        label = f"{sid[0]}/{sid[1]}"
        f_read = wholesheet.paths(folder, sid)[0]
        if not f_read.is_file():
            refused.append(f"{label}: no stored read ({f_read}); read it with "
                           f"`sfzmaps find read --run {run} --sheet {label}`")
        why = wholesheet.check(sid)
        if why:
            refused.append(why)
        blocks = frame_data.WORK / sid[0] / sid[1] / "blocks_px.geojson"
        if not blocks.is_file():
            refused.append(f"{label}: no traced blocks ({blocks})")
        try:
            era = pages.source(sid[0]).era
        except (KeyError, OSError) as e:
            refused.append(f"{label}: no era: {e.args[0] if isinstance(e, KeyError) else e}")
            continue
        if era in eras:
            refused.append(f"{label}: the same era ({era}) as {eras[era]}; give one sheet per era")
        eras.setdefault(era, label)
    if refused:
        for why in refused:
            typer.echo(why, err=True)
        typer.echo("nothing measured", err=True)
        raise typer.Exit(1)
    out = {"run": run, "trial": trial, "offset_means": OFFSET_MEANS, "eras": {}}
    for sid in sids:
        label = f"{sid[0]}/{sid[1]}"
        street = read.load(wholesheet.paths(folder, sid)[0])[0]
        img = cv2.imread(str(frame_data.page_path(*sid, work=frame_data.WORK)), cv2.IMREAD_UNCHANGED)  # as runner
        keep = pages.load_keep(sid, img.shape[:2])
        if street.shape != keep.shape:
            typer.echo(f"{label}: its read is {street.shape}, its page {keep.shape}; read it again", err=True)
            raise typer.Exit(1)
        with open(frame_data.WORK / sid[0] / sid[1] / "blocks_px.geojson") as f:
            traced = [shape(ft["geometry"]) for ft in json.load(f)["features"]]
        ink = linework.binarize(img)
        half = linework.stroke_width(ink) / 2.0
        one = np.asarray(placeblocks.one_to_one(areas.areas(street, keep), traced), bool)
        seeds = placeblocks.anchors(ink, keep, photo=linework.is_photo(img), scale=placeblocks.tracing_scale(traced))
        dark = np.asarray(placeblocks.mostly_ink(ink, traced, anchors=seeds), bool)

        def offset(mask):
            return placeblocks.outline_offset(street, keep, [p for p, m in zip(traced, mask) if m])

        got = offset(one)
        v = np.rint(placeblocks.offsets(street, keep, [p for p, m in zip(traced, one) if m])).astype(int)
        era = pages.source(sid[0]).era
        out["eras"][era] = {
            "sheet": label, "offset": got,
            "offset_measured_on": "traced blocks alone in their finder area (one to one)",
            "traced_blocks": len(traced), "one_to_one_blocks": int(one.sum()),
            "mostly_ink_blocks": int(dark.sum()),
            "offset_all_blocks": offset(np.ones(len(traced), bool)),
            "offset_one_to_one_mostly_white": offset(one & ~dark),
            "offset_one_to_one_mostly_ink": offset(one & dark),
            "offset_points_by_whole_px": {str(k): int(n) for k, n in zip(*np.unique(v, return_counts=True))},
            "half_stroke_px": half}
        typer.echo(f"{era}: {label}: {len(traced)} traced blocks, {int(one.sum())} alone in their finder area; "
                   f"offset (past the traced outline, px) median {got['median_px']:.2f} "
                   f"(p25 {got['p25_px']:.2f}, p75 {got['p75_px']:.2f}) over {got['points']} outline points; "
                   f"half line width {half:.2f}")
    f = trials.TRIAL / trial / "outline_offsets.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1) + "\n")
    typer.echo(str(f))


@find_app.command("blocks")
def find_blocks(run: str = typer.Option(..., "--run", help="The run whose stored whole-sheet reads make the blocks "
                                                          "(data/find/runs/<run>/sheets/)."),
                trial: str = typer.Option(..., "--trial", help="The trial's name: the blocks go into its finder arm, "
                                                              "data/find/trial/<trial>/finder/."),
                sheet: list[str] = typer.Option(..., "--sheet", help="A sheet, as <series>/<sheet> (repeatable).")
                ) -> None:
    """Make each sheet's blocks for placement from the finder's read of it, and write them into the trial's finder arm
    in place of the traced blocks copied there (data/find/trial/<trial>/finder/work/<series>/<sheet>/
    blocks_px.geojson); the file's freshness record is left as copied, so the placement steps run there take the
    finder's blocks as current and do not trace the page again.  Today's data and the traced arm are only read.

    A block is a piece of ground the finder calls not street (its street chance at most one half); pieces touching the
    page's edge or the frame cut are cut, and each other piece is outlined by the tracing step's own outline code: an
    outlined block's sides are moved in by half the sheet's measured line width, a solid one's are not, by the tracing
    step's own rule.  Each block keeps how sure the read was of the street round it.

    For each sheet it writes the counts to data/find/trial/<trial>/blocks/<series>/<sheet>.json and two debugging
    pictures at the page's own size, each with a reduced overview copy (.small.png), in
    data/find/trial/<trial>/look/<series>/: <sheet>__outlines.png, today's traced blocks (from the traced arm) beside
    the finder's, each block shaded with a bright outline; and <sheet>__areas.png, each area the finder calls not
    street tinted by what became of it (alone, joined to other blocks, cut at the rim, dropped by the cleaning).  It
    prints one line: today's traced blocks and good corners; the finder's areas, those cut at the rim, those cleaned
    away, its blocks and good corners; how many of its areas hold two or more of the year's blocks (today's lots by
    block, placed on the page by today's placement); how many of today's traced blocks lie alone in a finder area,
    share one with other traced blocks, or lie on ground it calls street; and today's blocks inside the frame cut by
    what became of the area they fall in.  Exit status
    1, writing nothing, when a sheet has no finder arm root (naming `sfzmaps find trial root`), no stored read (naming
    `sfzmaps find read`), no traced arm root, no frame cut or no placement today, or is a duplicate scan.  $0."""
    import numpy as np
    from shapely.geometry import shape

    from sfzmaps.find import areas, look, pages, placeblocks, read, trial as trials, wholesheet
    from sfzmaps.frame import data as frame_data
    for name, hint in ((run, "--run"), (trial, "--trial")):
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise typer.BadParameter(f"{name!r}: one folder name (no path separator, no '..')", param_hint=hint)
    sids = []
    for s in sheet:
        ser, _, name = s.partition("/")
        if not ser or not name:
            raise typer.BadParameter(f"{s}: expected <series>/<sheet>", param_hint="--sheet")
        sids.append((ser, name))
    root, traced_root = trials.arm_root(trial, "finder"), trials.arm_root(trial, "traced")
    folder = pages.OUT / "runs" / run
    refused = []
    for sid in sids:
        label = f"{sid[0]}/{sid[1]}"
        work = root / "work" / sid[0] / sid[1]
        missing = [f for f in ("page.png", "blocks_px.geojson", "blocks_px.geojson.inputs.json")
                   if not (work / f).is_file()]
        if missing:
            refused.append(f"{label}: the finder arm root has no {', '.join(missing)} ({work}); make it with "
                           f"`sfzmaps find trial root --trial {trial} --arm finder --sheet {label}`")
        f_read = wholesheet.paths(folder, sid)[0]
        if not f_read.is_file():
            refused.append(f"{label}: no stored read ({f_read}); read it with "
                           f"`sfzmaps find read --run {run} --sheet {label}`")
        why = wholesheet.check(sid)
        if why:
            refused.append(why)
        f_traced = traced_root / "work" / sid[0] / sid[1] / "blocks_px.geojson"
        if not f_traced.is_file():
            refused.append(f"{label}: no traced arm root ({f_traced}); make it with "
                           f"`sfzmaps find trial root --trial {trial} --arm traced --sheet {label}`")
        placed = config.DATA_DIR / "warp" / sid[0] / sid[1] / "transform.json"
        if not placed.is_file():
            refused.append(f"{label}: no placement today ({placed}), to put the year's blocks on the page")
    if refused:
        for why in refused:
            typer.echo(why, err=True)
        typer.echo("no blocks written", err=True)
        raise typer.Exit(1)
    for sid in sids:
        label = f"{sid[0]}/{sid[1]}"
        work = root / "work" / sid[0] / sid[1]
        f_traced = traced_root / "work" / sid[0] / sid[1] / "blocks_px.geojson"
        lum = frame_data.load_luminance(work / "page.png")
        points = placeblocks.lot_block_points(sid, lum.shape)
        got = placeblocks.write_sheet(sid, folder, root, f_traced, points, run=run)
        with open(work / "blocks_px.geojson") as f:
            found = json.load(f)
        with open(f_traced) as f:
            traced = json.load(f)

        def corners(d):
            return np.array([[np.nan if v is None else v for v in c] for c in d.get("corners") or []],
                            float).reshape(-1, 2)
        pic = look.outlines_picture(
            lum, [shape(ft["geometry"]) for ft in traced["features"]], corners(traced),
            [shape(ft["geometry"]) for ft in found["features"]], corners(found),
            f"RESULT (debugging picture): {sid[0]} / {sid[1]}: today's traced blocks (left) and the finder's blocks "
            f"from run {run} (right)")
        look_dir = trials.TRIAL / trial / "look" / sid[0]
        look_dir.mkdir(parents=True, exist_ok=True)
        f_pic, f_small = look_dir / f"{sid[1]}__outlines.png", look_dir / f"{sid[1]}__outlines.small.png"
        cv2.imwrite(str(f_pic), pic)
        cv2.imwrite(str(f_small), look.reduced(pic))
        keep = pages.load_keep(sid, lum.shape)
        lab = areas.areas(read.load(wholesheet.paths(folder, sid)[0])[0], keep)
        kinds = placeblocks.area_kinds(lab, placeblocks.edge_cut(lab, keep),
                                       [ft["properties"]["area"] for ft in found["features"]], points)
        pic = look.areas_picture(lum, lab, kinds, f"RESULT (debugging picture): {sid[0]} / {sid[1]}: what became of "
                                                  f"each area the finder calls not street (run {run})")
        f_areas, f_areas_small = look_dir / f"{sid[1]}__areas.png", look_dir / f"{sid[1]}__areas.small.png"
        cv2.imwrite(str(f_areas), pic)
        cv2.imwrite(str(f_areas_small), look.reduced(pic))
        f_counts = trials.TRIAL / trial / "blocks" / sid[0] / f"{sid[1]}.json"
        f_counts.parent.mkdir(parents=True, exist_ok=True)
        f_counts.write_text(json.dumps({"run": run, "trial": trial, "sheet": label, **got,
                                        "blocks_file": str(work / "blocks_px.geojson"), "picture": str(f_pic),
                                        "overview": str(f_small), "areas_picture": str(f_areas),
                                        "areas_overview": str(f_areas_small)}, indent=1) + "\n")
        typer.echo(f"{label}: today {got['traced_blocks']} traced blocks, {got['traced_good_corners']} good corners; "
                   f"finder {got['areas']} areas, {got['edge_cut']} cut at the rim, {got['dropped_by_cleaning']} "
                   f"cleaned away, {got['blocks']} blocks ({got['moved_in']} moved in by half the "
                   f"{got['stroke_px']:g} px line), {got['good_corners']} good corners; "
                   f"{got['areas_with_two_or_more']} areas hold two or more of the year's blocks "
                   f"({got['blocks_in_them']} of the {got['lot_blocks_on_page']} on the page); today's traced blocks: "
                   f"{got['traced_one_to_one']} alone in a finder area, {got['traced_sharing']} sharing "
                   f"{got['traced_sharing_areas']} areas, {got['traced_on_street']} on street; today's blocks inside "
                   f"the frame cut by what became of them: {got['lot_blocks_alone']} alone, {got['lot_blocks_joined']} "
                   f"joined, {got['lot_blocks_cut_at_rim']} cut at the rim, {got['lot_blocks_dropped']} dropped by "
                   f"cleaning, {got['lot_blocks_on_street']} on street")
        typer.echo(f"  {f_counts}\n  {f_pic}\n  {f_small}\n  {f_areas}\n  {f_areas_small}")


@trial_app.command("root")
def find_trial_root(trial: str = typer.Option(..., "--trial", help="The trial's name: data/find/trial/<trial>/."),
                    arm: str = typer.Option(..., "--arm", help="traced (today's blocks) or finder (the finder's)."),
                    sheet: list[str] = typer.Option(..., "--sheet", help="A sheet, as <series>/<sheet> "
                                                                         "(repeatable).")) -> None:
    """Make one arm's data root, data/find/trial/<trial>/<arm>/: raw, ref and cache linked to today's, and each sheet's
    work files up to and including its blocks (page.png, page.json, labels.json, blocks_px.geojson and the paper warp,
    with their freshness records and saved answers) copied, never linked, into its work/<series>/<sheet>/.  Nothing
    made after the blocks is copied, so the placement steps run there (SFZMAPS_DATA_DIR=<root>) make it again.
    The root is made whole beside its place and renamed into it, so a failed copy leaves nothing.  Prints each file
    copied.  Exit status 1, writing nothing, when the root is today's data root or lies in its work, warp or out
    folder, already holds a link into today's data (other than raw, ref and cache) or anything else, a sheet is a duplicate scan, has a review or a seed (its placement does not start from its
    blocks) or lacks one of those files, or the root was prepared before (delete its folder to redo it).  Reads
    nothing else, $0."""
    from sfzmaps.find import trial as trials
    if arm not in trials.ARMS:
        raise typer.BadParameter(f"{arm!r}: one of {', '.join(trials.ARMS)}", param_hint="--arm")
    sids = []
    for s in sheet:
        ser, _, name = s.partition("/")
        if not ser or not name:
            raise typer.BadParameter(f"{s}: expected <series>/<sheet>", param_hint="--sheet")
        sids.append((ser, name))
    try:
        root = trials.arm_root(trial, arm)
        copied = trials.prepare(root, sids, source=config.DATA_DIR)
    except (ValueError, FileExistsError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    for f in copied:
        typer.echo(str(f))


@trial_app.command("numbers")
def find_trial_numbers(trial: str = typer.Option(..., "--trial", help="The trial's name: data/find/trial/<trial>/."),
                       run: str = typer.Option("all-pieces", "--run", help="The finder run whose training ground is "
                                                                           "measured (data/find/runs/<run>/).")
                       ) -> None:
    """For every sheet in the trial's arm roots, print one line per data root (today's data, then each arm that is
    there): its blocks, good corners, control points, the placement check with its held-out errors, and the warp's
    chosen correction with its share of block-side points within 1 px of a printed line, each read from that root's
    own files.  Then whether the traced arm came out exactly as today, file by file, and the run's training ground on
    the sheet (the share under its lot-labelled sheets, and its hand crops there).  Reads today's and the trial's
    data; writes only data/find/trial/<trial>/numbers.json.  $0."""
    from sfzmaps.find import pages, trial as trials
    if not run or run == "." or ".." in run or "/" in run or "\\" in run:
        raise typer.BadParameter(f"{run!r}: a run's name is one folder name (no path separator, no '..')",
                                 param_hint="--run")
    try:
        sids = trials.trial_sheets(trial)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--trial") from None
    if not sids:
        typer.echo(f"{trials.TRIAL / trial}: no arm root with a sheet; make one with `sfzmaps find trial root`",
                   err=True)
        raise typer.Exit(1)
    folder = pages.OUT / "runs" / run
    if not (folder / "log.json").is_file():
        typer.echo(f"{folder / 'log.json'}: no run log", err=True)
        raise typer.Exit(1)
    out = {"trial": trial, "run": run, "sheets": {}}
    for sid in sids:
        got = {"today": trials.numbers(config.DATA_DIR, sid)}
        typer.echo(trials.row(sid, "today", got["today"]))
        for arm in trials.ARMS:
            root = trials.arm_root(trial, arm)
            if (root / "work" / sid[0] / sid[1]).is_dir():
                got[arm] = trials.numbers(root, sid)
                typer.echo(trials.row(sid, arm, got[arm]))
        if "traced" in got:
            got["reproduced"] = trials.reproduced(trials.arm_root(trial, "traced"), config.DATA_DIR, sid)
            typer.echo(f"{sid[0]}/{sid[1]} traced as today: " +
                       ", ".join(f"{k} {'same' if v else 'DIFFERS'}" for k, v in got["reproduced"].items()))
        try:
            got["trained_ground"] = ground = trials.trained_ground(sid, folder, config.DATA_DIR)
        except FileNotFoundError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from None
        parts = ", ".join(f"{k} {v:.3f}" for k, v in ground["by_sheet"].items())
        typer.echo(f"{sid[0]}/{sid[1]} trained ground ({run}): {ground['lot_labelled_share']:.3f} of its ground under "
                   f"lot-labelled training sheets" + (f" ({parts})" if parts else "") + "; hand crops on it: "
                   + (", ".join(ground["hand_crops"]) or "none"))
        out["sheets"][f"{sid[0]}/{sid[1]}"] = got
    f = trials.TRIAL / trial / "numbers.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    typer.echo(str(f))


@trial_app.command("compare")
def find_trial_compare(trial: str = typer.Option(..., "--trial", help="The trial's name: data/find/trial/<trial>/."),
                       run: str = typer.Option("all-pieces", "--run", help="The finder run whose training ground is "
                                                                           "measured (data/find/runs/<run>/).")
                       ) -> None:
    """Compare each trial sheet placed from today's traced blocks (the traced arm) with it placed from the finder's
    blocks (the finder arm), beside today's placement.

    First it checks every sheet's finder blocks file is the one `sfzmaps find blocks` wrote (not traced again or
    changed since), and that today and both arms have the sheet placed (control points and a warped scan).  Then, per
    sheet and for today, traced and finder: the numbers of `find trial numbers`; how far the sheet's block-side
    points (every 2 ft along every side of the year's blocks, the points the warp scores) lie from the nearest
    control point, in feet (median and 90th percentile); and the share of control points on ground the run was
    trained on with lot labels.

    It picks three places on each sheet, the same for both arms, from today's placement: the worst (the most
    block-side points more than 1 px from the print, the warp's own measure), the busiest (the most block-side
    points) and the centre (the sheet's middle), each 1,200 px of today's warped scan on a side.  For each it writes a
    picture: the scan warped onto today's map, placed from today's traced blocks on the left and from the finder's
    blocks on the right, at the warped scan's own resolution, today's block lines drawn on top in magenta, 3 px wide.
    These numbers and pictures are reports: nothing here passes or fails a sheet.

    Places that fall in the same box get one picture, named after the first of them, and the printed row says which
    places share it.  When no block-side point is off the print today, the worst place is left out and the row says
    so.

    Writes only data/find/trial/<trial>/compare.json and data/find/trial/<trial>/look/<series>/<sheet>__<place>.png
    (with a reduced .small.png copy when a picture is larger than 3,000 px; its own earlier place pictures are
    removed first); today's data and both arms are only read.  Prints the table.  Exit status 1, writing nothing, when
    the data root in force lies inside the trial (unset SFZMAPS_DATA_DIR), a finder blocks file was traced again or
    changed, or a sheet is not yet placed in today's data or an arm (naming the step to run).  $0."""
    import numpy as np

    from sfzmaps.find import look, pages, trial as trials
    if not run or run == "." or ".." in run or "/" in run or "\\" in run:
        raise typer.BadParameter(f"{run!r}: a run's name is one folder name (no path separator, no '..')",
                                 param_hint="--run")
    try:
        sids = trials.trial_sheets(trial)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--trial") from None
    if not sids:
        typer.echo(f"{trials.TRIAL / trial}: no arm root with a sheet; make one with `sfzmaps find trial root`",
                   err=True)
        raise typer.Exit(1)
    roots = {"today": config.DATA_DIR, **{arm: trials.arm_root(trial, arm) for arm in trials.ARMS}}
    here = (trials.TRIAL / trial).resolve()
    if config.DATA_DIR.resolve() == here or here in config.DATA_DIR.resolve().parents:
        typer.echo(f"{config.DATA_DIR}: the data root in force lies inside the trial ({here}), so it is not today's "
                   f"data; unset SFZMAPS_DATA_DIR and run again", err=True)
        raise typer.Exit(1)
    refused = []
    for sid in sids:
        try:
            trials.check_blocks(trial, sid)
        except ValueError as e:
            refused.append(str(e))
    for sid in sids:
        for label, root in roots.items():
            refused += trials.unplaced(root, sid, label)
    folder = pages.OUT / "runs" / run
    if not (folder / "log.json").is_file():
        refused.append(f"{folder / 'log.json'}: no run log")
    if refused:
        for why in refused:
            typer.echo(why, err=True)
        typer.echo("nothing compared", err=True)
        raise typer.Exit(1)
    try:
        ground = trials.trained_union(folder, config.DATA_DIR)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    what = {"worst": "the most block-side points off the print today", "busiest": "the most block-side points",
            "centre": "the sheet's middle"}
    out = {"trial": trial, "run": run, "window_px": trials.WINDOW_PX, "line_px": trials.LINE_PX, "sheets": {}}
    for sid in sids:
        label = f"{sid[0]}/{sid[1]}"
        pts = trials.side_points(sid)
        got = {"numbers": {}, "reach_ft": {}, "control_points_on_trained_ground": {}, "side_points": len(pts),
               "places": {}}
        for col, root in roots.items():
            gcps = trials.control_points(root, sid)
            got["numbers"][col] = n = trials.numbers(root, sid)
            got["reach_ft"][col] = reach = trials.reach_ft(pts, gcps)
            got["control_points_on_trained_ground"][col] = on = trials.share_on(gcps, ground)

            def ft(v):
                return "-" if v is None else f"{v:.1f}"
            typer.echo(trials.row(sid, col, n) + f"; block-side points to the nearest control point p50 "
                       f"{ft(reach['p50'])} / p90 {ft(reach['p90'])} ft; control points on trained ground "
                       + ("-" if on is None else f"{on:.3f}"))
        boxes = trials.places(sid, config.DATA_DIR)
        off = np.asarray(trials.off_print(sid, config.DATA_DIR), bool)
        lines = trials.block_lines(list(boxes.values()))
        look_dir = trials.TRIAL / trial / "look" / sid[0]
        look_dir.mkdir(parents=True, exist_ok=True)
        for place in trials.PLACES:                 # this command's own pictures of an earlier run, never kept stale
            for end in (".png", ".small.png"):
                (look_dir / f"{sid[1]}__{place}{end}").unlink(missing_ok=True)
        if "worst" not in boxes:
            got["places"]["worst"] = {"left_out": "no block-side point is off the print today"}
            typer.echo(f"{label} worst: left out, no block-side point is off the print today")
        for first, group in trials.shared(boxes).items():
            b = boxes[first]
            inside = trials.in_box(pts, b)
            pic = trials.placed_picture(
                trials.arm_root(trial, "traced") / "warp" / sid[0] / sid[1] / "warped.tif",
                trials.arm_root(trial, "finder") / "warp" / sid[0] / sid[1] / "warped.tif", b, lines,
                f"RESULT: {label}, {' = '.join(group)}: left placed from today's traced blocks, right from the "
                f"finder's blocks")
            f_pic = look_dir / f"{sid[1]}__{first}.png"
            f_small = trials.write_picture(pic, f_pic)
            for place in group:
                entry = {"box_ft": list(b), "chosen_by": what[place], "side_points": int(inside.sum()),
                         "off_print_today": int((inside & off).sum()), "picture": str(f_pic),
                         "picture_px": [int(pic.shape[1]), int(pic.shape[0])], "same_box_as": group}
                if f_small is not None:
                    entry["overview"] = str(f_small)
                got["places"][place] = entry
            names = " and ".join(f"{p} ({what[p]})" for p in group)
            typer.echo(f"{label} {names}" + (" share one box" if len(group) > 1 else "") + f": {int(inside.sum())} "
                       f"block-side points, {int((inside & off).sum())} off the print today; {f_pic} "
                       f"({pic.shape[1]} x {pic.shape[0]} px)")
        out["sheets"][label] = got
    f = trials.TRIAL / trial / "compare.json"
    f.write_text(json.dumps(out, indent=1))
    typer.echo(str(f))


if __name__ == "__main__":
    app()
