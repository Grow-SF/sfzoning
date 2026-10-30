import math

from sfzmaps import config


def test_estimate_tokens_768_tile():
    # 768x768 -> 24x24 = 576 patches x 1.2 multiplier
    assert config.estimate_image_tokens(768, 768) == math.ceil(576 * 1.2) == 692


def test_estimate_tokens_applies_patch_budget():
    # 2048x2048 high detail is shrunk to <= 2500 patches (docs worked example: 3000 tokens)
    assert config.estimate_image_tokens(2048, 2048) == 3000


def test_estimate_tokens_never_enlarges():
    assert config.estimate_image_tokens(32, 32) == math.ceil(1.2)


def test_model_and_prices():
    assert config.VISION_MODEL == "gpt-6.1-sol"
    assert (config.PRICE_IN_PER_M, config.PRICE_OUT_PER_M) == (2.0, 10.0)
    assert config.VISION_CONCURRENCY == 8


def test_estimate_tokens_fits_2048_box_before_patch_budget():
    # 4000x1000 -> fit 2048 box -> 2048x512 = 64x16 = 1024 patches -> 1229 tokens
    assert config.estimate_image_tokens(4000, 1000) == math.ceil(1024 * 1.2)


def test_estimate_includes_prompt_text_tokens(tmp_path, monkeypatch):
    """R11: every uncached call is charged EST_TEXT_TOKENS_PER_CALL text tokens too."""
    import re

    import cv2
    import numpy as np
    from typer.testing import CliRunner

    from sfzmaps import cli, llm, pages
    from sfzmaps.pages import Page

    png = tmp_path / "page.png"
    cv2.imwrite(str(png), np.full((600, 600), 255, np.uint8))
    monkeypatch.setattr(pages, "discover", lambda: [Page("1948", "1", png, None)])
    monkeypatch.setattr(cli, "_page_png", lambda page: png)
    monkeypatch.setattr(llm, "is_cached", lambda *a: False)
    out = CliRunner().invoke(cli.app, ["estimate", "--year", "1948"]).output
    calls = int(re.search(r"calls: (\d+)", out).group(1))
    img = int(re.search(r"image tokens: ([\d,]+)", out).group(1).replace(",", ""))
    txt = int(re.search(r"prompt text tokens: ([\d,]+)", out).group(1).replace(",", ""))
    usd = float(re.search(r"\$([\d.]+)", out).group(1))
    assert calls >= 1 and txt == calls * config.EST_TEXT_TOKENS_PER_CALL
    want = ((img + txt) * config.PRICE_IN_PER_M + calls * config.EST_OUT_TOKENS * config.PRICE_OUT_PER_M) / 1e6
    assert abs(usd - want) < 0.006 and "estimate" in out
