"""
Segmentation module
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from cellpose import io, models

logger = logging.getLogger(__name__)


class CellposeSegmenter:
    """Load a (custom-pretrained) Cellpose model and run segmentation."""

    def __init__(
        self,
        pretrained_model: str,
        gpu: bool = True,
        niter: int = 1000,
    ) -> None:
        """
        Args:
            pretrained_model: Path to the pretrained Cellpose checkpoint file.
            gpu: Use GPU if available.
            niter: Number of Cellpose iterations per image.
        """
        io.logger_setup()
        self.niter = niter
        logger.info("Loading Cellpose model from %s …", pretrained_model)
        self.model = models.CellposeModel(
            pretrained_model=pretrained_model, gpu=gpu
        )
        logger.info("Model loaded.")


    def segment(
        self,
        images: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], list, list]:
        """
        Run segmentation on a list of RGB uint8 images.

        Returns:
            masks_pred : list of HxW label arrays (0 = background, 1..K = objects)
            flows      : list of raw Cellpose flow fields
            styles     : list of style vectors (one per image)
        """
        logger.info("Segmenting %d image(s) …", len(images))
        masks_pred, flows, styles = self.model.eval(images, niter=self.niter)
        logger.info("Segmentation done.")
        return masks_pred, flows, styles

    @staticmethod
    def load_images(
        directory: str | Path,
        extension: str = ".jpg",
    ) -> Tuple[List[np.ndarray], List[Path]]:
        """
        Load all images with the given extension from *directory*.

        Returns:
            imgs  : list of np.ndarray images
            files : list of corresponding Path objects
        """
        from natsort import natsorted

        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = natsorted(
            [
                f
                for f in directory.glob(f"*{extension}")
                if "_masks" not in f.name and "_flows" not in f.name
            ]
        )
        if not files:
            raise FileNotFoundError(
                f"No '{extension}' images found in {directory}"
            )

        logger.info("Found %d image(s) in %s", len(files), directory)
        imgs = [io.imread(str(f)) for f in files]
        return imgs, files
