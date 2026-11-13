"""A sheet's key and folder.  Key: what follows the last '-p' in the folder name ('01' for 2000-digital-zoning-p01 or
20180102113004574-p01), else the whole name ('1' in 1960, '20180102094142210' in 1921).  Folder: the series' sheet
folder whose name is the key, or ends in '-p' + key (exactly one)."""
import os
from pathlib import Path


def key(sheet):
    return sheet.rsplit("-p", 1)[-1]


def sheet_dir(series, k, work=None):
    if work is None:
        from sfzmaps import config
        work = config.WORK_DIR
    names = [n for n in os.listdir(Path(work) / series) if n == k] or \
            [n for n in os.listdir(Path(work) / series) if n.endswith("-p" + k)]
    assert len(names) == 1, (series, k, names)
    return names[0]
