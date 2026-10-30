"""List and download public Google Drive folders."""
import html
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

LIST_URL = "https://drive.google.com/embeddedfolderview?id={id}"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download"

_ENTRY_RE = re.compile(r'id="entry-([^"]+)".*?flip-entry-title">([^<]+)<', re.DOTALL)


@dataclass(frozen=True)
class DriveEntry:
    id: str
    title: str
    is_folder: bool


def parse_listing(page: str) -> list[DriveEntry]:
    entries = []
    for m in _ENTRY_RE.finditer(page):
        entries.append(
            DriveEntry(
                id=m.group(1),
                title=html.unescape(m.group(2)),
                is_folder="/drive/folders/" in m.group(0),
            )
        )
    return entries


def list_folder(folder_id: str) -> list[DriveEntry]:
    r = httpx.get(LIST_URL.format(id=folder_id), follow_redirects=True, timeout=60)
    r.raise_for_status()
    return parse_listing(r.text)


def download(file_id: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    params = {"id": file_id, "export": "download", "confirm": "t"}
    try:
        with httpx.stream(
            "GET", DOWNLOAD_URL, params=params, follow_redirects=True, timeout=120
        ) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _safe_name(title: str) -> str:
    return title.replace("/", "_")


def sync(folder_id: str, dest: Path) -> list[Path]:
    """Mirror a Drive folder recursively into dest; returns paths of files present."""
    dest.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for entry in list_folder(folder_id):
        path = dest / _safe_name(entry.title)
        if entry.is_folder:
            files.extend(sync(entry.id, path))
            continue
        if not (path.exists() and path.stat().st_size > 0):
            download(entry.id, path)
        files.append(path)
    return files
