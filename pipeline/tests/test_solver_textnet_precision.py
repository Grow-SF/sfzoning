import dataclasses

import pytest
import torch

from sfzmaps.solver import paths
from sfzmaps.solver.textnet import data, predict, train
from test_solver_textnet import OLD, SMALL, maps  # noqa: F401  (the fixture)


def test_precision_is_a_recipe_choice():
    assert "precision=float16" in train.Recipe(precision="float16").name()
    assert "precision" not in train.Recipe().name()
    with pytest.raises(ValueError):
        train.Recipe(precision="half")


def _fit(maps, seed=0):
    sheets = data.load_sheets([paths.SERIES, OLD], hold_out_validation=True)
    recipe = dataclasses.replace(SMALL, iterations=3, precision="float16")
    return recipe, train.fit(sheets, recipe, device="cpu", seed=seed)


def test_float16_fit_is_repeatable(maps):
    _, a = _fit(maps)
    _, b = _fit(maps)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert all(v.dtype == torch.float32 for v in sa.values() if v.is_floating_point())


def test_saved_float16_model_loads(maps, tmp_path):
    recipe, net = _fit(maps)
    f = train.save_model(tmp_path / "m.pt", net, recipe, "cpu")
    loaded = predict.load_model(f, "cpu")
    for k, v in net.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])
