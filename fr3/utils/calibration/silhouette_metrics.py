"""Comparing an observed robot mask with a rendered one.

Masks are HxW images of the robot: observed (SAM2 on the camera image) and
rendered (the URDF meshes seen from a candidate camera pose).

Losses (torch, differentiable in the rendered silhouette ``rend`` in [0, 1]):

* ``area_loss``: sum over pixels of (obs - rend)^2. Counts the pixels where
  the two disagree; its gradient is the same size wherever an edge is wrong,
  and zero if the silhouettes do not overlap at all.
* ``boundary_loss``: every rendered pixel OUTSIDE the observed mask costs its
  distance (px) to the mask, every observed pixel the render MISSES costs its
  distance to the mask's outside. A rendered edge 10 px from the observed
  edge is pulled 10x harder than one 1 px away, and an edge that is off the
  mask still gets a pull towards it. It is the area between the two
  outlines, weighted by how far each piece is from the observed outline.

Metrics (numpy, on binary masks; for evaluating a result):

* ``iou``: intersection / union of the two masks (1 = identical).
* ``boundary_error``: distances (px) between the two outlines. For every
  outline pixel of one mask, the distance to the nearest outline pixel of the
  other, both ways. Outline pixels on the image border are skipped: there the
  robot is cut off by the frame, not by its own shape.
"""

from __future__ import annotations

import cv2
import numpy as np


def distance_maps(obs_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(d_out, d_in), float32 HxW, in pixels.

    d_out: distance from each pixel to the nearest mask pixel (0 inside).
    d_in:  distance from each mask pixel to the nearest non-mask pixel (0 outside).
    Pixels beyond the image count as unknown, not as background: the mask
    edge cut by the image border is not treated as an outline."""
    m = np.asarray(obs_mask, dtype=bool)
    d_out = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    d_in = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    return d_out.astype(np.float32), d_in.astype(np.float32)


def area_loss(rend, obs):
    """sum (obs - rend)^2 (torch tensors, HxW)."""
    return ((obs - rend) ** 2).sum()


def boundary_loss(rend, d_out, d_in):
    """sum rend * d_out + (1 - rend) * d_in (torch tensors, HxW).

    Zero only when the render covers exactly the observed mask."""
    return (rend * d_out).sum() + ((1.0 - rend) * d_in).sum()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def outline(mask: np.ndarray, skip_border: bool = True) -> np.ndarray:
    """Bool HxW: mask pixels with a non-mask 4-neighbour."""
    m = np.asarray(mask, dtype=bool)
    eroded = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    edge = m & ~eroded.astype(bool)
    if skip_border:
        edge[0, :] = edge[-1, :] = False
        edge[:, 0] = edge[:, -1] = False
    return edge


def boundary_error(obs_mask: np.ndarray, rend_mask: np.ndarray) -> dict:
    """Outline distances in pixels: ``mean`` and ``p95`` over both directions
    (obs -> rend and rend -> obs). inf if either outline is empty."""
    eo, er = outline(obs_mask), outline(rend_mask)
    if not eo.any() or not er.any():
        return {"mean": float("inf"), "p95": float("inf")}
    to_o = cv2.distanceTransform((~eo).astype(np.uint8), cv2.DIST_L2, 5)
    to_r = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 5)
    d = np.concatenate([to_o[er], to_r[eo]])
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95))}
