""" I/O utilities (visualisation, paths, logging setup).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def setup_logging(level: int = logging.INFO) -> None:
    """Configure root-level logging with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def show_image_grid(
    images: Sequence,
    titles: Optional[Sequence[str]] = None,
    cols: int = 5,
    cell_size: float = 3.0,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Display (or save) a grid of PIL images or numpy arrays.

    Args:
        images    : Sequence of PIL Images or numpy arrays.
        titles    : Optional per-image titles.
        cols      : Number of columns in the grid.
        cell_size : Size of each cell in inches.
        title     : Overall figure title.
        save_path : If set, save figure to this path instead of showing.
    """
    n = len(images)
    if n == 0:
        return
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cell_size * cols, cell_size * rows))
    axes_flat = np.array(axes).flatten()

    for i, img in enumerate(images):
        ax = axes_flat[i]
        ax.imshow(img)
        if titles:
            ax.set_title(str(titles[i]), fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def show_segmentation_pairs(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    save_dir: Optional[str | Path] = None,
) -> None:
    """
    Display side-by-side raw image / predicted mask for each sample.
    """
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, mask) in enumerate(zip(images, masks)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img.squeeze())
        axes[0].set_title(f"Image #{i}")
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
        axes[1].set_title(f"Predicted Mask #{i}")
        axes[1].axis("off")
        plt.tight_layout()

        if save_dir:
            plt.savefig(str(save_dir / f"seg_{i:04d}.png"), bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
