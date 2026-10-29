from sfzmaps import pointsfile

TEXT = """mapX,mapY,pixelX,pixelY,enable
6007144.08130914252251387,2121915.95729747088626027,1561.14845902719434889,-1211.33159701422960097,1
6007227.59995126444846392,2112256.62590355519205332,960.59996106385403891,-5709.1475338777718207,0
"""


def test_read_points_flips_y(tmp_path):
    p = tmp_path / "1.jpg.points"
    p.write_text("#CRS: EPSG:2227\n" + TEXT)
    g = pointsfile.read_points(p)
    assert len(g) == 2
    assert g[0].px == 1561.14845902719434889 and g[0].py == 1211.33159701422960097
    assert (g[0].wx, g[0].wy) == (6007144.08130914252251387, 2121915.95729747088626027)
    assert g[0].source == "answer_key" and g[0].enabled
    assert g[1].py == 5709.1475338777718207 and g[1].enabled is False


def test_read_points_scale(tmp_path):
    p = tmp_path / "1.pdf.points"
    p.write_text(TEXT)
    g = pointsfile.read_points(p, scale=2.0)
    assert g[0].px == 2 * 1561.14845902719434889
    assert g[0].py == 2 * 1211.33159701422960097
    assert g[0].wx == 6007144.08130914252251387
