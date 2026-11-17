"""The Mission Street review page (`wordread.review`) on small synthetic pages, boxes, reads and Terra words."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from sfzmaps.solver.wordread import measures, review

H, W = 120, 300


def _bars(xs, y0=52, y1=60, width=3):
    """Ink: a vertical bar `width` px wide at each x of `xs`, rows y0..y1 (a word's glyphs)."""
    ink = np.zeros((H, W), bool)
    for x in xs:
        ink[y0:y1 + 1, x:x + width] = True
    return ink


def _chance(value):
    return np.full((H, W), value, np.uint8)


WORD = np.array([[40, 50], [100, 50], [100, 62], [40, 62]], float)    # 60 px long, 12 tall, reading left to right
INK_WORD = _bars(range(44, 97, 8))
INK_WORD_ON_LINE = INK_WORD.copy()
INK_WORD_ON_LINE[56, :] = True                                        # a line through the word, the page's width
CHANCE_HIGH = _chance(255)
CHANCE_062 = _chance(158)                                             # 158 / 255 = 0.62
COMPONENTS_LINE = review.components(CHANCE_HIGH, INK_WORD_ON_LINE, 0.99)
COMPONENTS_NONE = review.components(CHANCE_062, INK_WORD, 0.99)
INK_SPLIT = _bars([44, 50, 56, 82, 88, 94])                           # two clusters 23 px apart: two components
COMPONENTS_SPLIT = review.components(CHANCE_HIGH, INK_SPLIT, 0.99)
COMPONENTS_WORD = review.components(CHANCE_HIGH, INK_WORD, 0.99)


def test_colour_rules():
    assert review.colour("RH-2", "RH-2") == "green" and review.colour("RM-2", "RH-2") == "purple"
    assert review.colour("RH-2", None) == "none"


def test_colour_compares_normalised_strings_and_reads_a_printed_form_as_its_code():
    assert review.colour("rh-2 ", "RH-2") == "green"
    assert review.colour("", "RH-2") == "purple"
    forms = {"24TH-MISSION NCD": "24TH MISSION"}
    assert review.colour("24TH MISSION", "24th-mission  NCD", forms) == "green"
    assert review.colour("24TH MISSION", "24TH-MISSION NCD") == "purple"


def test_colour_folds_a_trailing_full_stop_and_doubled_spaces():
    assert review.colour("ST", "ST.") == "green" and review.colour("ST.", "ST") == "green"
    assert review.colour("VALENCIA ST", "VALENCIA  ST.") == "green"
    assert review.colour("RH 2", "RH-2") == "purple" and review.colour("S.T", "ST") == "purple"


def test_red_is_ink_at_chance_half_or_more():
    chance = np.round(np.array([[0.0, 0.49, 0.5, 1.0]]) * 255).astype(np.uint8)
    ink = np.ones_like(chance, bool)
    assert review.text_ink_of(ink, chance).tolist() == [[False, False, True, True]]
    assert not review.text_ink_of(~ink, chance).any()                 # paper is never red


def test_a_word_inside_a_long_line_component_is_merged_into_line():
    r, d = review.miss_reason(WORD, CHANCE_HIGH, INK_WORD_ON_LINE, COMPONENTS_LINE, 0.99)
    assert r == "merged_into_line" and d["component_px"] > 3 * d["word_length_px"]


def test_a_word_below_the_level_says_how_close():
    r, d = review.miss_reason(WORD, CHANCE_062, INK_WORD, COMPONENTS_NONE, 0.99)
    assert r == "below_level" and 0.6 <= d["max_chance"] <= 0.63
    assert d["share_above"] == 0 and d["components"] == 0


def test_a_word_whose_ink_falls_in_two_components_is_split():
    r, d = review.miss_reason(WORD, CHANCE_HIGH, INK_SPLIT, COMPONENTS_SPLIT, 0.99)
    assert r == "split" and d["components"] == 2 and d["share_above"] == 1


def test_a_word_held_whole_by_one_short_component_is_wrong_size():
    r, d = review.miss_reason(WORD, CHANCE_HIGH, INK_WORD, COMPONENTS_WORD, 0.99)
    assert r == "wrong_size" and d["components"] == 1 and d["component_px"] <= 3 * d["word_length_px"]


def test_a_word_off_the_map_is_outside_map_and_the_reading_is_kept():
    reading = {"top": "RH-2", "p_top": 0.94}
    r, d = review.miss_reason(WORD, CHANCE_HIGH, INK_WORD, COMPONENTS_WORD, 0.99, on_map=False, reading=reading)
    assert r == "outside_map" and d["reading"] == reading


def test_a_raw_label_image_gives_the_same_reason_as_components():
    r, d = review.miss_reason(WORD, CHANCE_HIGH, INK_WORD_ON_LINE, COMPONENTS_LINE.labels, 0.99)
    assert r == "merged_into_line" and d["component_px"] == COMPONENTS_LINE.length[d["component"]]


def test_a_word_merged_with_its_neighbour_names_it():
    ink = _bars(range(44, 97, 8)) | _bars([100, 108, 116])            # 5 px apart: one component holds both words
    terra = pd.DataFrame([{"sheet": "s", "text": t, "kind": "zoning code", "quarter": 1,
                           **dict(zip(measures.C_COLS, _poly(*r)))}
                          for t, r in (("RH-2", (40, 50, 96, 62)), ("12", (99, 50, 120, 62)))])
    got = review.miss_reasons(terra, CHANCE_HIGH, ink, {"nw": 0.5, "ne": 0.99, "sw": 0.5, "se": 0.5},
                              np.ones(2, bool), [0, 1])
    assert got[0][0] == got[1][0] == "merged_with_word"
    assert got[0][1]["merged_with"] == ["12"] and got[1][1]["merged_with"] == ["RH-2"]
    assert got[0][1]["shares_component_with"] == ["12"] and got[1][1]["shares_component_with"] == ["RH-2"]
    assert got[0][1]["level"] == 0.99                                  # the word's quarter's level


def test_two_words_joined_by_one_component_are_merged_with_word_before_line_or_level():
    ink = INK_WORD_ON_LINE | _bars([150, 158, 166])                    # a second word on the same long line
    comps = review.components(CHANCE_HIGH, ink, 0.99)
    other = np.array([[[146, 50], [172, 50], [172, 62], [146, 62]]], float)
    r, d = review.miss_reason(WORD, CHANCE_HIGH, ink, comps, 0.99, others=other, other_names=["12"], on_map=False)
    assert r == "merged_with_word" and d["merged_with"] == ["12"]


def test_a_word_on_a_long_line_with_no_other_word_on_it_is_merged_into_line():
    ink = INK_WORD_ON_LINE | _bars([204, 212, 220], 92, 100)           # another word, off the line
    comps = review.components(CHANCE_HIGH, ink, 0.99)
    other = np.array([[[200, 90], [226, 90], [226, 102], [200, 102]]], float)
    r, d = review.miss_reason(WORD, CHANCE_HIGH, ink, comps, 0.99, others=other, other_names=["12"])
    assert r == "merged_into_line" and d["merged_with"] == []


def test_a_neighbours_rectangle_overlapping_the_word_is_not_a_merge():
    other = np.array([[[90, 50], [130, 50], [130, 62], [90, 62]]], float)    # overlaps the word's last glyph; no ink
    r, d = review.miss_reason(WORD, CHANCE_HIGH, INK_WORD, COMPONENTS_WORD, 0.99, others=other, other_names=["x"])
    assert r == "wrong_size" and d["merged_with"] == []


def test_the_mission_sheet_holds_the_longest_stretch_of_mission_street():
    streets = gpd.GeoDataFrame({"streetname": ["MISSION ST", "MISSION ST", "VALENCIA ST"]},
                               geometry=[shapely.LineString([(0, 0), (100, 0)]),
                                         shapely.LineString([(100, 0), (100, 30)]),
                                         shapely.LineString([(200, 0), (200, 500)])], crs="EPSG:2227")
    footprints = {"a": shapely.box(-10, -10, 40, 10),           # 40 ft of the first piece
                  "b": shapely.box(70, -10, 110, 40),           # 30 + 30 ft
                  "c": shapely.box(150, -10, 250, 600)}         # a long street, but not Mission
    assert review.mission_sheet(streets, footprints) == "b"


# --- a tiny page: one matched box, one box Terra missed (text), one junk box (none), one Terra word we missed ------

def _poly(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _tiny():
    ink = _bars([12, 20, 28], 12, 20) | _bars([72, 80, 88], 12, 20) | _bars([132, 140, 148], 12, 20) \
        | _bars([192, 200, 208], 12, 20)
    ink[90, :] = True                                          # a line, not text
    chance = np.zeros((H, W), np.uint8)
    chance[5:30, :] = 255                                      # the words' row is text
    rects = {"A": (8, 8, 34, 24), "B": (68, 8, 94, 24), "C": (128, 8, 154, 24)}
    boxes = pd.DataFrame([{"sheet": "s", "quarter": 0, "axis": 0.0, "length": 26.0, "height": 16.0,
                           "mean_chance": 1.0, "ink_px": 81, "level": 0.99, "p_zoning": 0.9, "p_block": 0.05,
                           "p_street": 0.03, "p_other": 0.02, **dict(zip(measures.G_COLS, _poly(*r)))}
                          for r in rects.values()])
    reads = {0: {"closed": {"RH-2": 0.9, "RM-1": 0.1}, "top": "RH-2", "text": "RH-2", "p_180": 0.1,
                 "open": {"RH-2": 0.88, "other": 0.1, "none": 0.02}},
             1: {"closed": {"RH-2": 0.6, "RM-1": 0.4}, "top": "other", "text": "3751", "p_180": 0.0},
             2: {"closed": {"RH-2": 0.5, "RM-1": 0.5}, "top": "none", "text": "", "p_180": 0.7}}
    terra = pd.DataFrame([{"sheet": "s", "text": t, "kind": k, "quarter": 0, **dict(zip(measures.C_COLS, _poly(*r)))}
                          for t, k, r in (("RH-2", "zoning code", (8, 8, 34, 24)),
                                          ("RC-3", "zoning code", (188, 8, 214, 24)))])
    return review.page_data("s", boxes, reads, terra, chance, None, ink=ink,
                            levels={q: 0.99 for q in ("nw", "ne", "sw", "se")}, on_map=np.ones(len(terra), bool),
                            terra_reads={1: {"top": "RC-3", "p_top": 0.94}}, forms={}, models=["synthetic"])


PAGE_DATA_TINY = _tiny()


def test_page_data_sorts_the_boxes_and_the_missed_words():
    d = PAGE_DATA_TINY
    by_row = {b["row"]: b for b in d["boxes"]}
    assert by_row[0]["status"] == "matched" and by_row[0]["colour"] == "green" and by_row[0]["shown"] == "RH-2"
    assert by_row[1]["status"] == "text" and by_row[1]["shown"] == "3751"      # the open best path, not a code
    assert by_row[2]["status"] == "none"
    (m,) = d["missed"]
    assert m["text"] == "RC-3" and m["reason"] in review.REASONS and m["details"]["reading"]["top"] == "RC-3"
    b = d["numbers"]["boxes"]["0.5"]
    assert (b["terra"], b["ours"], b["matched"], b["missed"]) == (2, 3, 1, 1)
    assert sum(b["missed_reasons"].values()) == 1
    assert d["numbers"]["reads"]["all"] == {"matched": 1, "same": 1, "differs": 0}


def test_shading_never_covers_red_ink(tmp_path):
    out = review.render(PAGE_DATA_TINY, tmp_path / "p.html")
    assert review.red_px_after_overlay(out) == review.red_px_of(PAGE_DATA_TINY)


def test_shading_tints_the_paper_in_green_and_yellow_boxes_and_keeps_the_ink(tmp_path):
    img = review.page_image(review.render(PAGE_DATA_TINY, tmp_path / "p.html"))
    assert tuple(img[10, 70]) == review.COLOURS["green_paper"]          # paper inside the green box (ours, B)
    assert tuple(img[10, 190]) == review.COLOURS["yellow_paper"]        # paper inside Terra's missed word
    assert tuple(img[15, 73]) == review.COLOURS["text_ink"]             # text ink inside the green box
    assert tuple(img[90, 120]) == review.COLOURS["other_ink"]           # the line
    assert tuple(img[60, 120]) == review.COLOURS["paper"]


def test_the_change_since_the_run_before_is_shown(tmp_path):
    before = json.loads(json.dumps(PAGE_DATA_TINY["numbers"]))
    before["boxes"]["0.5"]["matched"] = 3
    d = dict(PAGE_DATA_TINY, delta=measures.compare(PAGE_DATA_TINY["numbers"], before))
    assert d["delta"]["boxes"]["0.5"]["matched"] == -2
    assert "vs last run: matched <b>-2</b>" in review.render(d, tmp_path / "p.html").read_text()


def test_the_page_is_self_contained(tmp_path):
    html = review.render(PAGE_DATA_TINY, tmp_path / "p.html").read_text()
    assert html.startswith("<title>") and "<html" not in html and "<body" not in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "") and "https://" not in html
    assert 'name="claude:viewport-fit-sides"' in html and "prefers-color-scheme: dark" in html
