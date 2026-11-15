"""The sheet warp's fit device and its deterministic building blocks, against what the fits compute today
(`F.grid_sample` on the CPU, as in `first.py` and `probabilistic.py`, and scipy's Gaussian filter).

The blocks run on `fit_device()`: the CPU on CI, the Mac's GPU there unless SFZMAPS_WARP_DEVICE says otherwise."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from sfzmaps.sheetwarp.device import MapSampler, PointField, _blur_by_convolution, blur, fit_device


@pytest.fixture
def device():
    return fit_device()


def todays_field(G, points, spacing):
    """the correction field at the points as the fits compute it today: grid_sample over the grid's extent"""
    gh, gw = G.shape[-2:]
    ext = torch.tensor([(gw - 1) * spacing, (gh - 1) * spacing], dtype=torch.float32)
    xy = torch.as_tensor(points)
    g = torch.stack([xy[:, 0] / ext[0] * 2 - 1, xy[:, 1] / ext[1] * 2 - 1], -1)[None, None]
    return F.grid_sample(G, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :].T


def todays_sample(maps, bins, xy):
    """each point's value on the map of its direction as the probabilistic fit samples it today: grid_sample per
    direction, zero off the map"""
    B, _, h, w = maps.shape
    g = torch.stack([xy[:, 0] / (w - 1) * 2 - 1, xy[:, 1] / (h - 1) * 2 - 1], -1)
    vals, idx = [], []
    for b in range(B):
        ii = torch.as_tensor(np.flatnonzero(bins == b))
        if len(ii):
            vals.append(F.grid_sample(maps[b:b + 1], g[ii][None, None], mode="bilinear", padding_mode="zeros",
                                      align_corners=True)[0, 0, 0, :])
            idx.append(ii)
    return torch.zeros(xy.shape[0]).index_copy(0, torch.cat(idx), torch.cat(vals))


def test_point_field_matches_grid_sample(device):
    rng = np.random.default_rng(1)
    gh, gw, spacing = 7, 10, 50
    width, height = (gw - 1) * spacing, (gh - 1) * spacing
    scattered = rng.uniform([-80, -80], [width + 80, height + 80], (4000, 2))     # about half of them off the grid
    on_nodes = np.stack(np.meshgrid(np.arange(gw) * spacing, np.arange(gh) * spacing), -1).reshape(-1, 2)
    points = np.concatenate([scattered, on_nodes]).astype(np.float32)             # the far edges' nodes included
    G = torch.from_numpy(rng.normal(0, 1, (1, 2, gh, gw)).astype(np.float32))
    upstream = rng.normal(0, 1, (len(points), 2)).astype(np.float32)

    grid_today = G.clone().requires_grad_()
    u_today = todays_field(grid_today, points, spacing)
    (u_today * torch.from_numpy(upstream)).sum().backward()

    grid = G.to(device).requires_grad_()
    u = PointField(points, (gh, gw), spacing, device).field(grid)
    (u * torch.from_numpy(upstream).to(device)).sum().backward()

    assert u.shape == (len(points), 2)
    assert np.abs(u.detach().cpu().numpy() - u_today.detach().numpy()).max() <= 1e-6
    grad_today = grid_today.grad.numpy()
    assert np.abs(grid.grad.cpu().numpy() - grad_today).max() <= 1e-5 * np.abs(grad_today).max()


def test_point_field_repeats_bit_for_bit(device):
    rng = np.random.default_rng(2)
    gh, gw, spacing = 12, 16, 50
    points = rng.uniform([0, 0], [(gw - 1) * spacing, (gh - 1) * spacing], (200_000, 2)).astype(np.float32)
    G = torch.from_numpy(rng.normal(0, 2, (1, 2, gh, gw)).astype(np.float32)).to(device)
    upstream = torch.from_numpy(rng.normal(0, 1, (len(points), 2)).astype(np.float32)).to(device)

    def run():          # about a thousand contributions per grid node: summed in any other order, the bits change
        grid = G.clone().requires_grad_()
        u = PointField(points, (gh, gw), spacing, device).field(grid)
        (u * upstream).sum().backward()
        return u.detach().cpu(), grid.grad.cpu()

    u1, grad1 = run()
    u2, grad2 = run()
    assert torch.equal(u1, u2)
    assert torch.equal(grad1, grad2)


def test_map_sampler_matches_grid_sample(device):
    rng = np.random.default_rng(3)
    B, h, w, n = 18, 45, 70, 6000
    centres = (rng.random((B, h, w)) < 0.03).astype(np.float32)
    maps = torch.from_numpy(np.stack([gaussian_filter(c, 2.0) for c in centres])[:, None])  # like a stage's maps
    xy = rng.uniform([-3, -3], [w + 2, h + 2], (n, 2)).astype(np.float32)             # some off the map
    bins = rng.integers(0, B - 3, n)                                                  # some directions without points
    # the bilinear surface has a kink at every whole pixel: right on one, either side's slope is as good
    frac = xy - np.floor(xy)
    keep = ((frac > 1e-3) & (frac < 1 - 1e-3)).all(1)
    xy, bins = xy[keep], bins[keep]
    upstream = rng.normal(0, 1, len(xy)).astype(np.float32)

    xy_today = torch.from_numpy(xy).requires_grad_()
    v_today = todays_sample(maps, bins, xy_today)
    (v_today * torch.from_numpy(upstream)).sum().backward()

    xy_new = torch.from_numpy(xy).to(device).requires_grad_()
    v = MapSampler(bins, h, w, device).sample(maps.to(device), xy_new)
    (v * torch.from_numpy(upstream).to(device)).sum().backward()

    assert v.shape == (len(xy),)
    v = v.detach().cpu().numpy()
    assert np.abs(v - v_today.detach().numpy()).max() <= 1e-6
    off_map = (xy[:, 0] <= -1) | (xy[:, 0] >= w) | (xy[:, 1] <= -1) | (xy[:, 1] >= h)
    assert off_map.sum() > 100 and (v[off_map] == 0).all()
    grad_today = xy_today.grad.numpy()
    assert np.abs(xy_new.grad.cpu().numpy() - grad_today).max() <= 1e-5 * np.abs(grad_today).max()


# a last stage, a middle one, and the widest a fit starts at: half the widest street logged (33 px); its kernel
# (66 px each way) reaches past the map's edges.  The float32 sums' error grows with the kernel: at sigma 30 both
# OpenCV's and the GPU's reach 1e-6 of the peak.
@pytest.mark.parametrize("sigma", [1.2, 7.5, 16.5])
def test_blur_matches_scipy(sigma, device):
    rng = np.random.default_rng(4)
    B, h, w = 3, 40, 60
    weights = ((rng.random((B, h, w)) < 0.02) * rng.random((B, h, w))).astype(np.float16)   # as block_line_weights
    expected = np.stack([gaussian_filter(np.asarray(m, np.float32), sigma) for m in weights])
    blurred = {
        "on the fit device": blur(weights, sigma, device),
        "on the fit device, the weights uploaded once": blur(torch.from_numpy(weights).to(device), sigma, device),
        "on the CPU (OpenCV)": blur(weights, sigma, torch.device("cpu")),
        "by convolution on the CPU (the GPU's arithmetic)": _blur_by_convolution(torch.from_numpy(weights), sigma),
    }
    for how, maps in blurred.items():
        assert maps.shape == (B, 1, h, w) and maps.dtype == torch.float32, how
        assert np.abs(maps[:, 0].cpu().numpy() - expected).max() <= 1e-6 * expected.max(), how


def test_fit_device_honours_the_override(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", "cpu")
    assert fit_device() == torch.device("cpu")
    monkeypatch.delenv("SFZMAPS_WARP_DEVICE")
    assert fit_device() == torch.device("mps")             # the Mac's GPU when there is one
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert fit_device() == torch.device("cpu")             # else the CPU
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", "mps")
    assert fit_device() == torch.device("mps")
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", "gpu")
    with pytest.raises(ValueError, match="SFZMAPS_WARP_DEVICE"):
        fit_device()
