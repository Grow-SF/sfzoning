"""The merged page's ink as pen strokes, for finding leader lines: nothing judged, nothing removed.

The ink (`page.load_ink()`, raw: text and lines alike) is thinned to centre lines (`skimage.morphology.skeletonize`).
A node is a run of touching centre-line px that don't have exactly two centre-line neighbours (line ends and
junctions); a stroke is the centre line between two nodes, or a closed loop with no node (both its ends one node).
Each stroke keeps its path, the ink's width along it (twice the distance from the centre line to the paper), its
length, and the number of strokes at each end node.  A short dead end, shorter than the stroke's median width
(a junction's bump or a serif), is kept and tagged a spur.

Writes, under data/solver/labels/:
  strokes.npz  xy      float32 (points, 2): every stroke's path in map ft (EPSG:2227), stroke after stroke; a point
                       is a grid px centre, `grid.to_map(col + 0.5, row + 0.5)`, and the first and last are its end
                       nodes' centres
               w       float32 per point: the ink's width there, ft
               off     int64 (strokes + 1): stroke i's points are xy[off[i]:off[i + 1]]
               a, b    int64 per stroke: its end nodes
               length  float32 per stroke: its length along the path, ft
               wmed    float32: its median width, ft
               wiqr    float32: the spread of its width (upper less lower quartile), ft
               spur    bool: a dead end (an end node with this stroke alone) shorter than its median width
               dega, degb  int64: the number of strokes at node a, and at node b (a loop counts twice)
               Placement (`solver/placement.py`) reads xy, off and wmed.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from sfzmaps.solver import grid, page, paths

# a centre-line px's eight neighbours (row, column steps), in the order a stroke is followed
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def strokes(ink: np.ndarray) -> dict[str, np.ndarray]:
    """The strokes of `ink` (bool, True = ink, on the grid from its top-left corner): strokes.npz's arrays."""
    ink = np.asarray(ink, bool)
    height, width = ink.shape
    half = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    line = skeletonize(ink)
    ring = np.ones((3, 3), np.float32)
    ring[1, 1] = 0
    degree_px = cv2.filter2D(line.view(np.uint8), -1, ring, borderType=cv2.BORDER_CONSTANT)
    is_node = line & (degree_px != 2)
    del degree_px
    n_labels, label = cv2.connectedComponents(is_node.view(np.uint8), connectivity=8)
    ys, xs = np.nonzero(label)
    k = label[ys, xs]
    count = np.maximum(np.bincount(k, minlength=n_labels), 1)
    node_xy = list(zip((np.bincount(k, xs, n_labels) / count).tolist(), (np.bincount(k, ys, n_labels) / count).tolist()))

    def neighbours(y, x):
        return [(y + dy, x + dx) for dy, dx in NEIGHBOURS
                if 0 <= y + dy < height and 0 <= x + dx < width and line[y + dy, x + dx]]

    def new_node(y, x):
        node_xy.append((float(x), float(y)))
        return len(node_xy) - 1

    visited = np.zeros_like(line)
    found = []                                         # (a, b, path [(row, col)])
    for y, x in zip(*(v.tolist() for v in np.nonzero(is_node))):
        a = int(label[y, x])
        for q in neighbours(y, x):
            if is_node[q]:
                b = int(label[q])
                if a < b:                              # two nodes touching: a stroke of their two px
                    found.append((a, b, [(y, x), q]))
                continue
            if visited[q]:
                continue
            path, prev, cur = [(y, x), q], (y, x), q
            visited[q] = True
            end = None
            while True:
                ahead = [p for p in neighbours(*cur) if p != prev and p not in path[-3:]]
                hit = [p for p in ahead if is_node[p] and not (label[p] == a and len(path) < 3)]
                if hit:
                    path.append(hit[0])
                    end = int(label[hit[0]])
                    break
                ahead = [p for p in ahead if not visited[p] and not is_node[p]]
                if not ahead:
                    break
                prev, cur = cur, ahead[0]
                visited[cur] = True
                path.append(cur)
            if end is None:                            # ran into px already followed: a new end node there
                end = new_node(*cur)
            found.append((a, end, path))
    for y, x in zip(*(v.tolist() for v in np.nonzero(line & ~visited & ~is_node))):   # loops with no node
        if visited[y, x]:
            continue
        a = new_node(y, x)
        path, prev, cur = [(y, x)], None, (y, x)
        visited[y, x] = True
        while True:
            ahead = [p for p in neighbours(*cur) if p != prev and not visited[p]]
            if not ahead:
                break
            prev, cur = cur, ahead[0]
            visited[cur] = True
            path.append(cur)
        path.append((y, x))
        found.append((a, a, path))

    degree = np.zeros(len(node_xy), np.int64)
    for a, b, _ in found:
        degree[a] += 1
        degree[b] += 1
    points, widths, off = [], [], [0]
    out = {key: [] for key in ("a", "b", "length", "wmed", "wiqr", "spur", "dega", "degb")}
    for a, b, path in found:
        p = np.vstack([node_xy[a], np.array([(x, y) for y, x in path], float), node_xy[b]])
        w = 2 * half[np.clip(p[:, 1].astype(int), 0, height - 1), np.clip(p[:, 0].astype(int), 0, width - 1)] \
            * grid.FT_PER_PX
        length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) * grid.FT_PER_PX
        median = np.median(w)
        q1, q3 = np.percentile(w, [25, 75])
        points.append(p)
        widths.append(w)
        off.append(off[-1] + len(p))
        out["a"].append(a)
        out["b"].append(b)
        out["length"].append(length)
        out["wmed"].append(float(median))
        out["wiqr"].append(float(q3 - q1))
        out["spur"].append(a != b and (degree[a] == 1 or degree[b] == 1) and length < median)
        out["dega"].append(degree[a])
        out["degb"].append(degree[b])
    xy = grid.to_map(np.vstack(points) + 0.5) if points else np.zeros((0, 2))
    return {"xy": xy.astype(np.float32),
            "w": (np.concatenate(widths) if widths else np.zeros(0)).astype(np.float32),
            "off": np.array(off, np.int64),
            "a": np.array(out["a"], np.int64), "b": np.array(out["b"], np.int64),
            "length": np.array(out["length"], np.float32), "wmed": np.array(out["wmed"], np.float32),
            "wiqr": np.array(out["wiqr"], np.float32), "spur": np.array(out["spur"], bool),
            "dega": np.array(out["dega"], np.int64), "degb": np.array(out["degb"], np.int64)}


def build_strokes() -> dict:
    """Find the strokes of the merged page's ink and write strokes.npz into LABELS_DIR.  Returns how many strokes
    (and spurs) and their length in miles."""
    s = strokes(page.load_ink())
    paths.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.LABELS_DIR / "strokes.npz", **s)
    return {"strokes": len(s["length"]), "spurs": int(s["spur"].sum()), "nodes": int(len(np.union1d(s["a"], s["b"]))),
            "miles": round(float(s["length"].astype(float).sum()) / 5280, 1)}
