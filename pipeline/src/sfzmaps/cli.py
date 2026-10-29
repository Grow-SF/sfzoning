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
