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
