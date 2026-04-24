"""
Cutout extraction module.
Cuts individual particles out of a microscopy image using Cellpose label masks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Type alias for a single cutout result
Cutout = Tuple[int, Image.Image, Optional[Tuple[int, int, int, int]]]


def cutouts_from_label_mask(
    im: np.ndarray,
    label_mask: np.ndarray,
    crop_to_mask: bool = True,
    min_pixels: int = 100,
) -> List[Cutout]:
    """
    Extract individual RGBA particle cutouts from a label mask.

    Args:
        im          : HxWx3 uint8 RGB image.
        label_mask  : HxW integer array where 0 = background,
                      positive integers identify object instances.
        crop_to_mask: If True, crop each cutout to its bounding box.
        min_pixels  : Discard objects whose mask area is below this threshold.

    Returns:
        List of (obj_id, cutout_rgba, bbox) tuples.
        *cutout_rgba* is a PIL RGBA image; *bbox* is (left, upper, right, lower)
        or None when crop_to_mask is False.
    """
    im_rgba = Image.fromarray(im).convert("RGBA")

    ids = np.unique(label_mask)
    ids = ids[ids != 0]  # drop background

    results: List[Cutout] = []
    for obj_id in ids:
        bin_mask = (label_mask == obj_id).astype(np.uint8) * 255
        if np.count_nonzero(bin_mask) < min_pixels:
            logger.debug("Object %d skipped (<%d px)", obj_id, min_pixels)
            continue

        alpha = Image.fromarray(bin_mask).convert("L")
        cutout = im_rgba.copy()
        cutout.putalpha(alpha)

        bbox = None
        if crop_to_mask:
            bbox = alpha.getbbox()
            if bbox is None:
                continue
            cutout = cutout.crop(bbox)

        results.append((int(obj_id), cutout, bbox))

    logger.info(
        "Extracted %d cutout(s) (min_pixels=%d)", len(results), min_pixels
    )
    return results


def save_cutouts(
    cutouts: List[Cutout],
    save_dir: str | Path,
    prefix: str = "obj",
) -> List[Path]:
    """
    Save RGBA cutouts as PNG files.

    Args:
        cutouts  : Output of :func:`cutouts_from_label_mask`.
        save_dir : Target directory (created if missing).
        prefix   : Filename prefix.

    Returns:
        List of saved file paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for obj_id, cutout_rgba, _ in cutouts:
        path = save_dir / f"{prefix}_{obj_id}.png"
        cutout_rgba.save(str(path))
        saved.append(path)

    logger.info("Saved %d cutout(s) to %s", len(saved), save_dir)
    return saved
