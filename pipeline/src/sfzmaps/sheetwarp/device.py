"""Where the sheet warp's fits run, and the pieces of them that must give the same answer every run there.

  fit_device   the Mac's GPU (PyTorch's `mps`) when there is one, else the CPU; SFZMAPS_WARP_DEVICE=cpu|mps overrides
  PointField   the correction field at a fixed set of points: the same values as `F.grid_sample` with border padding,
               as the fits read the grid today, but as fixed bilinear weights (u = W G) whose gradient into the grid
               (Wᵀ du) is summed per grid node in a fixed order.  `grid_sample`'s own backward adds every point's
               share onto its nodes with atomic adds on the GPU, in whatever order the threads run: the bits change
               from run to run.
  MapSampler   each point's value on the density map of its direction (zero off the map), as `F.grid_sample` with
               zero padding gives it, by one gather of its 4 neighbouring pixels; the gradient in the point's
               position is the bilinear surface's own slope (the maps take none), so nothing is scattered.
  blur         a stage's density maps: each direction's block-line weights blurred by a Gaussian, as
               `scipy.ndimage.gaussian_filter` does it (OpenCV on the CPU, a separable convolution on the GPU).
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

DEVICES = ("cpu", "mps")


def fit_device() -> torch.device:
    """The device the fits run on: SFZMAPS_WARP_DEVICE (cpu or mps) when set, else the Mac's GPU when there is one,
    else the CPU."""
    chosen = os.environ.get("SFZMAPS_WARP_DEVICE")
    if chosen:
        if chosen not in DEVICES:
            raise ValueError(f"SFZMAPS_WARP_DEVICE={chosen!r}: expected one of {', '.join(DEVICES)}")
        return torch.device(chosen)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PointField:
    """The correction field at a fixed set of points (crop px).  The grid's nodes are `spacing` px apart, the first
    at the crop's top-left; each point's value is the bilinear interpolation of its 4 surrounding nodes, and beyond
    the grid the value at its nearest edge.  Where each point falls on the grid is worked out once, in the float32
    arithmetic `F.grid_sample(..., align_corners=True, padding_mode="border")` uses on the points normalised by the
    grid's extent (as the fits did), so the values are the same to float rounding.

    `field(G)` is differentiable in G: the gradient into each node is the sum of its own contributions, in a fixed
    order, over a fixed padded table of them (nodes grouped by how many contributions they take, so the padding stays
    under half), never added up with scatters or atomic adds."""

    def __init__(self, points: np.ndarray, grid_shape: tuple[int, int], spacing: float, device):
        gh, gw = (int(v) for v in grid_shape)
        self.grid_shape, self.device = (gh, gw), torch.device(device)
        xy = np.asarray(points, np.float32).reshape(-1, 2)
        # grid_sample's arithmetic, in float32: normalised by the extent to [-1, 1], back to node units, clamped
        last_node = np.array([gw - 1, gh - 1], np.float32)
        extent = np.array([(gw - 1) * spacing, (gh - 1) * spacing], np.float32)
        normalised = xy / extent * np.float32(2) - np.float32(1)
        on_grid = np.clip((normalised + np.float32(1)) * (last_node / np.float32(2)), np.float32(0), last_node)
        node_before = np.floor(on_grid)                       # the node above-left of the point
        past = on_grid - node_before                          # how far past it, 0..1
        before = np.float32(1) - past
        col, row = node_before[:, 0].astype(np.int64), node_before[:, 1].astype(np.int64)
        next_col, next_row = np.minimum(col + 1, gw - 1), np.minimum(row + 1, gh - 1)   # a weight of 0 at the edge
        # the 4 corners in grid_sample's order (nw, ne, sw, se) and their weights, each (4, n)
        corner_nodes = np.stack([row * gw + col, row * gw + next_col, next_row * gw + col, next_row * gw + next_col])
        corner_weights = np.stack([before[:, 1] * before[:, 0], before[:, 1] * past[:, 0],
                                   past[:, 1] * before[:, 0], past[:, 1] * past[:, 0]])
        self.corner_nodes = torch.from_numpy(corner_nodes).to(self.device)
        self.corner_weights = torch.from_numpy(corner_weights).to(self.device)
        self.contributions = self._contributions_per_node(corner_nodes.ravel(), gh * gw)

    def _contributions_per_node(self, node_of_entry: np.ndarray, n_nodes: int) -> list[tuple[torch.Tensor, ...]]:
        """For the gradient: each node's entries (corner * n + point) in a fixed order (by corner, then point), as
        (nodes, table) per group of nodes with about as many entries (up to the next power of two); a table's rows
        are padded with the index of a zero added after the last entry."""
        order = np.argsort(node_of_entry, kind="stable")
        count = np.bincount(node_of_entry, minlength=n_nodes)
        start = np.cumsum(count) - count
        node_sorted = node_of_entry[order]
        place = np.arange(len(order)) - start[node_sorted]       # each entry's place among its node's
        size_class = np.ceil(np.log2(np.maximum(count, 1))).astype(int)
        row_of_node = np.zeros(n_nodes, np.int64)
        groups = []
        for c in np.unique(size_class[count > 0]):
            nodes = np.flatnonzero((count > 0) & (size_class == c))
            row_of_node[nodes] = np.arange(len(nodes))
            table = np.full((len(nodes), int(count[nodes].max())), len(order), np.int64)
            in_group = size_class[node_sorted] == c
            table[row_of_node[node_sorted[in_group]], place[in_group]] = order[in_group]
            groups.append((torch.from_numpy(nodes).to(self.device), torch.from_numpy(table).to(self.device)))
        return groups

    def field(self, G: torch.Tensor) -> torch.Tensor:
        """G (1, 2, gh, gw) on the device -> the field at the points (n, 2)"""
        if tuple(G.shape) != (1, 2, *self.grid_shape):
            raise ValueError(f"grid of shape {tuple(G.shape)}, expected {(1, 2, *self.grid_shape)}")
        return _FieldAtPoints.apply(G, self)

    def _interpolate(self, G: torch.Tensor) -> torch.Tensor:
        """(2, n): the weighted corners added in grid_sample's order"""
        nodes = G.reshape(2, -1)
        u = nodes[:, self.corner_nodes[0]] * self.corner_weights[0]
        for k in range(1, 4):
            u = u + nodes[:, self.corner_nodes[k]] * self.corner_weights[k]
        return u

    def _into_grid(self, du: torch.Tensor) -> torch.Tensor:
        """du (n, 2) -> Wᵀ du (2, gh * gw), each node's contributions summed in their fixed order"""
        shares = (du.T[:, None, :] * self.corner_weights[None]).reshape(2, -1)       # (2, 4 n), corner-major
        shares = torch.cat([shares, shares.new_zeros(2, 1)], 1)                      # the padding's zero
        dG = shares.new_zeros(2, self.grid_shape[0] * self.grid_shape[1])
        for nodes, table in self.contributions:
            dG[:, nodes] = shares[:, table].sum(-1)
        return dG


class _FieldAtPoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, G, point_field):
        ctx.point_field, ctx.grid_shape = point_field, G.shape
        return point_field._interpolate(G).T

    @staticmethod
    @once_differentiable
    def backward(ctx, du):
        return ctx.point_field._into_grid(du).reshape(ctx.grid_shape), None


class MapSampler:
    """Each point's value on the map of its own direction (`bins`, one per point) among B maps of h x w px:
    bilinear between the 4 pixels around it, zero off the map, as `F.grid_sample(..., padding_mode="zeros",
    align_corners=True)` gives it.  The point is read where it is in map px (grid_sample's normalisation and back
    only adds rounding), so values agree with grid_sample's to that rounding.  `sample(maps, xy)` is differentiable
    in xy, by the slope of the bilinear surface at each point; the maps take no gradient."""

    def __init__(self, bins: np.ndarray, h: int, w: int, device):
        self.h, self.w, self.device = int(h), int(w), torch.device(device)
        bins = np.asarray(bins, np.int64)
        self.n_maps_needed = int(bins.max()) + 1 if len(bins) else 0
        self.map_start = torch.from_numpy(bins * (self.h * self.w)).to(self.device)   # where its map starts, flat

    def sample(self, maps: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """maps (B, 1, h, w), or the same flattened, on the device; xy (n, 2) map px -> (n,)"""
        flat = maps.reshape(-1)
        if flat.numel() % (self.h * self.w) or flat.numel() // (self.h * self.w) < self.n_maps_needed:
            raise ValueError(f"maps of {flat.numel()} values: expected at least {self.n_maps_needed} of "
                             f"{self.h} x {self.w}")
        return _SampleMaps.apply(xy, flat, self)

    def _bilinear(self, flat: torch.Tensor, xy: torch.Tensor):
        """the value at each point and its slope in x and y"""
        h, w = self.h, self.w
        x, y = xy[:, 0], xy[:, 1]
        left, top = torch.floor(x), torch.floor(y)
        past_x, past_y = x - left, y - top
        before_x, before_y = 1 - past_x, 1 - past_y
        col, row = left.long(), top.long()

        def pixel(c, r):
            """the point's map at pixel (c, r), zero off the map"""
            on_map = (c >= 0) & (c < w) & (r >= 0) & (r < h)
            return torch.where(on_map, flat[self.map_start + r.clamp(0, h - 1) * w + c.clamp(0, w - 1)], 0)

        nw, ne, sw, se = pixel(col, row), pixel(col + 1, row), pixel(col, row + 1), pixel(col + 1, row + 1)
        value = (nw * (before_y * before_x) + ne * (before_y * past_x)
                 + sw * (past_y * before_x) + se * (past_y * past_x))
        slope_x = before_y * (ne - nw) + past_y * (se - sw)
        slope_y = before_x * (sw - nw) + past_x * (se - ne)
        return value, slope_x, slope_y


class _SampleMaps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, flat, sampler):
        value, slope_x, slope_y = sampler._bilinear(flat, xy)
        ctx.save_for_backward(slope_x, slope_y)
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, dvalue):
        slope_x, slope_y = ctx.saved_tensors
        return torch.stack([dvalue * slope_x, dvalue * slope_y], -1), None, None


def _gaussian_kernel(sigma: float) -> tuple[int, np.ndarray]:
    """scipy's: truncated at 4 sigma, radius int(4 sigma + 0.5), normalised to sum 1 (float64)"""
    radius = int(4.0 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return radius, k / k.sum()


def blur(maps, sigma: float, device) -> torch.Tensor:
    """Each map (B, h, w: an array, or a tensor already on `device` so the weights are uploaded once) blurred by a
    Gaussian of spread `sigma` px, as `scipy.ndimage.gaussian_filter(map, sigma)` gives it (truncated at 4 sigma, the
    edge reflected with the edge pixel repeated).  The sums are float32, so the error grows with the kernel: within
    1e-6 of the map's peak for the spreads the fits use (up to half the widest street, 16.5 px), about that at 30 px.
    OpenCV's separable filter on the CPU, a separable convolution on any other device.  -> (B, 1, h, w) float32 on
    `device`."""
    device = torch.device(device)
    if device.type == "cpu":
        source = maps.numpy() if isinstance(maps, torch.Tensor) else np.asarray(maps)
        _, k = _gaussian_kernel(sigma)
        out = np.empty((source.shape[0], 1) + source.shape[1:], np.float32)
        for b in range(source.shape[0]):
            out[b, 0] = cv2.sepFilter2D(np.asarray(source[b], np.float32), cv2.CV_32F, k, k,
                                        borderType=cv2.BORDER_REFLECT)
        return torch.from_numpy(out)
    on_device = maps if isinstance(maps, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(maps))
    return _blur_by_convolution(on_device.to(device), sigma)


def _reflected(n: int, radius: int) -> np.ndarray:
    """indices 0..n-1 extended by `radius` each way as scipy's 'reflect' mode extends them (d c b a | a b c d | d c b
    a), however far past the edge"""
    i = np.mod(np.arange(-radius, n + radius), 2 * n)
    return np.where(i < n, i, 2 * n - 1 - i)


def _blur_by_convolution(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """(B, h, w) on any device -> (B, 1, h, w) float32 there: down the columns, then along the rows, as scipy"""
    radius, k = _gaussian_kernel(sigma)
    _, h, w = maps.shape
    dev = maps.device
    k = torch.from_numpy(k.astype(np.float32)).to(dev)
    x = maps.float()[:, None]
    x = F.conv2d(x.index_select(2, torch.from_numpy(_reflected(h, radius)).to(dev)), k.view(1, 1, -1, 1))
    return F.conv2d(x.index_select(3, torch.from_numpy(_reflected(w, radius)).to(dev)), k.view(1, 1, 1, -1))
