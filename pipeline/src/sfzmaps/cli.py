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
    usd = tokens * in_price + calls * out_tokens * out_price
    typer.echo(f"pages: {npages}")
    typer.echo(f"calls: {calls} (cache hits: {hits})")
    typer.echo(f"image tokens: {tokens:,.0f}")
    typer.echo(f"projected cost: ${usd:,.2f}  (${config.PRICE_IN_PER_M:g}/M in, ${config.PRICE_OUT_PER_M:g}/M out, {out_tokens} output tokens/call)")
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
    for s in sorted({p.series for p in allp if all_ or p.series == year}):
        runner.run_series(s, sheet=sheet, until=until, force_auto=force_auto, force=force,
                          ref=ref, pages_list=allp)
        if sheet is None and until is None:  # a partial run must not republish the series
            _publish(s)


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
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)


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
