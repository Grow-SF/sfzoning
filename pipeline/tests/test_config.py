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
