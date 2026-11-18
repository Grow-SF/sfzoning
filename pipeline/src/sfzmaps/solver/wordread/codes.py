"""Code spotting's classes: each Terra word of a 2000 sheet as the class the detector's code head learns
(`train.Recipe.heads="words_code"`, `model.CODE`, `data.crop_targets(..., code=True)`).

The classes (`classes`, `model.CODE_CLASSES` of them): 0 "background" (no word), 1 "other text" (a word that is no
code), then the 2000 map's codes in the code list's order (`textread.read.CodeList`: its printed forms, each a code's).
A word is a code's when its string as the reader learns it (`textread.crops.norm_text`: upper case, single spaces) is
one of that code's printed forms; any other word (a block number, a street, a word in no form) is "other text".  Only
the 2000 sheets get classes (`prepare`): what another map's words would be on the 2000 map is not known, so their code
targets are not known either."""
from __future__ import annotations

import dataclasses

import numpy as np

from sfzmaps.solver import paths
from sfzmaps.solver.textnet import boxes, data, model

BACKGROUND, OTHER = 0, 1
FIRST_CODE = 2


def classes(code_list=None) -> list[str]:
    """The class names, by class index: "background", "other text", then the code list's codes (`CodeList.load()` when
    None).  ValueError when their number is not `model.CODE_CLASSES` (the code head's channels)."""
    if code_list is None:
        from sfzmaps.solver.textread.read import CodeList
        code_list = CodeList.load()
    out = ["background", "other text", *code_list.codes]
    if len(out) != model.CODE_CLASSES:
        raise ValueError(f"the code list gives {len(out)} classes, the code head has {model.CODE_CLASSES}")
    return out


def word_class(text, code_list) -> int:
    """A Terra word's class (the module) by its string."""
    from sfzmaps.solver.textread.crops import norm_text
    t = norm_text(text)
    if t is None or t not in code_list.forms:
        return OTHER
    return FIRST_CODE + int(code_list.code_of[code_list.forms.index(t)])


def prepare(sheets: list[data.SheetData], code_list=None, records: list[dict] | None = None) -> list[data.SheetData]:
    """The sheets with their words' classes (`SheetData.codes`, uint8 in the order of their words) on the 2000 sheets,
    the others unchanged; everything else shared (`dataclasses.replace`).  The words' strings are Terra's for the
    sheet (`boxes.terra_boxes`, the boxes `data.sheet_data` made the words from; `records` the saved answers, read
    when None): ValueError when they are not the sheet's words (another number or other rectangles)."""
    if code_list is None:
        from sfzmaps.solver.textread.read import CodeList
        code_list = CodeList.load()
    classes(code_list)
    if records is None and any(sd.series == paths.SERIES for sd in sheets):
        records = boxes.terra_records()
    out = []
    for sd in sheets:
        if sd.series != paths.SERIES:
            out.append(sd)
            continue
        found = boxes.terra_boxes(sd.series, sd.sheet, records)
        rects = data._corners([data.word_rect(b["box"], float(b.get("angle_deg") or 0.0)) for b in found])
        if rects.shape != sd.words.shape or not np.array_equal(rects, sd.words):
            raise ValueError(f"codes: {sd.series} {sd.sheet}: Terra's boxes are not the sheet's words "
                             f"({len(rects)} boxes, {len(sd.words)} words)")
        out.append(dataclasses.replace(sd, codes=np.array([word_class(b["text"], code_list) for b in found],
                                                          np.uint8)))
    return out


def counts(sheets: list[data.SheetData], areas=None) -> np.ndarray:
    """Each class's number of words on the sheets with classes (int64 (CODE_CLASSES,), background 0), every word, or
    with `areas` (a function of a sheet giving its words' mask, bool (n,)) only those it keeps."""
    n = np.zeros(model.CODE_CLASSES, np.int64)
    for sd in sheets:
        if not len(sd.codes):
            continue
        keep = np.ones(len(sd.codes), bool) if areas is None else areas(sd)
        n += np.bincount(sd.codes[keep], minlength=model.CODE_CLASSES)
    return n
