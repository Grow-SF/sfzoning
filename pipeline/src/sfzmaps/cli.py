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
        best = reading.calibrate_scale(_page_png(match[0]), series=year, report=rep)
    except (llm.LLMAuthError, llm.LLMRefusal, RuntimeError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    for s, r in rep.items():
        typer.echo(f"scale {s}: recall {r:.1%}")
    typer.echo(f"LABEL_SCALE[{year!r}] = {best}  (add to config.LABEL_SCALE)")


@app.command()
def estimate(
    year: str = typer.Option(None, "--year", help="Only this series."),
    all_: bool = typer.Option(False, "--all", help="All series."),
) -> None:
    """Project API calls, image tokens and USD for classify + label reading (no API calls)."""
    from sfzmaps import llm, pages, reading

    if not (year or all_):
        raise typer.BadParameter("pass --year Y or --all")
    in_price, out_price, out_tokens = 4 / 1e6, 20 / 1e6, 800
    calls = tokens = hits = npages = 0
    for page in pages.discover():
        if year and page.series != year:
            continue
        npages += 1
        png = cv2.imread(str(_page_png(page)), cv2.IMREAD_UNCHANGED)
        thumb, _ = reading.thumbnail(png)
        jobs = [thumb]
        # Mask boxes need Claude, so the estimate excludes by neatline only.
        skip = reading.exclusion_mask(png, None)
        _, tiles = reading.select_tiles(png, scale=config.label_scale(page.series), skip=skip)
        jobs += [t for _, _, t in tiles]
        for i, img in enumerate(jobs):
            prompt, schema = (reading.CLASSIFY_PROMPT, reading.PageInfo) if i == 0 \
                else (reading.LABEL_PROMPT, reading._Labels)
            if llm.is_cached(img, prompt, schema):
                hits += 1
                continue
            calls += 1
            tokens += img.shape[0] * img.shape[1] / 750
    usd = tokens * in_price + calls * out_tokens * out_price
    typer.echo(f"pages: {npages}")
    typer.echo(f"calls: {calls} (cache hits: {hits})")
    typer.echo(f"image tokens: {tokens:,.0f}")
    typer.echo(f"projected cost: ${usd:,.2f}  ($4/M in, $20/M out, {out_tokens} output tokens/call)")
    typer.echo("note: mask boxes are unknown without Claude, so tiles are excluded by neatline only (upper bound).")
