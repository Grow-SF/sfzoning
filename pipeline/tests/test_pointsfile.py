import pytest

from sfzmaps import pointsfile

TEXT = """mapX,mapY,pixelX,pixelY,enable
6007144.08130914252251387,2121915.95729747088626027,1561.14845902719434889,-1211.33159701422960097,1
6007227.59995126444846392,2112256.62590355519205332,960.59996106385403891,-5709.1475338777718207,0
"""


def test_read_points_flips_y_and_shifts_to_pixel_centers(tmp_path):
    p = tmp_path / "1.jpg.points"
    p.write_text("#CRS: EPSG:2227\n" + TEXT)
    g = pointsfile.read_points(p)
    assert len(g) == 2
    assert g[0].px == pytest.approx(1561.14845902719434889 - 0.5)
    assert g[0].py == pytest.approx(1211.33159701422960097 - 0.5)
    assert (g[0].wx, g[0].wy) == (6007144.08130914252251387, 2121915.95729747088626027)
    assert g[0].source == "answer_key" and g[0].enabled
    assert g[1].py == pytest.approx(5709.1475338777718207 - 0.5) and g[1].enabled is False


def test_read_points_corner_of_first_pixel_is_minus_half(tmp_path):
    p = tmp_path / "1.jpg.points"
    p.write_text("mapX,mapY,pixelX,pixelY,enable\n1,2,0,0,1\n1,2,1,-1,1\n")
    g = pointsfile.read_points(p, scale=2.0)
    assert (g[0].px, g[0].py) == (-0.5, -0.5)   # QGIS corner (0,0) -> our pixel 0's corner
    assert (g[1].px, g[1].py) == (1.5, 1.5)     # scaled first, then shifted


def test_read_points_scale(tmp_path):
    p = tmp_path / "1.pdf.points"
    p.write_text(TEXT)
    g = pointsfile.read_points(p, scale=2.0)
    assert g[0].px == pytest.approx(2 * 1561.14845902719434889 - 0.5)
    assert g[0].py == pytest.approx(2 * 1211.33159701422960097 - 0.5)
    assert g[0].wx == 6007144.08130914252251387


def test_read_points_wkt_crs_accepted(tmp_path):
    p = tmp_path / "1.jpg.points"
    p.write_text('#CRS: PROJCRS["NAD83 / California zone 3 (ftUS)",ID["EPSG",2227]]\n' + TEXT)
    assert len(pointsfile.read_points(p)) == 2


def test_read_points_wrong_crs_rejected(tmp_path):
    p = tmp_path / "1.jpg.points"
    p.write_text("#CRS: EPSG:4326\n" + TEXT)
    with pytest.raises(ValueError, match="EPSG:2227"):
        pointsfile.read_points(p)
