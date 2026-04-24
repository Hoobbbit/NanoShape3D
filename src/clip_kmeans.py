"""
Clustering module.
Embeds particle cutouts with CLIP, auto-selects K via silhouette score,
and finds the representative (closest-to-centroid) image per cluster.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import CLIPImageProcessor, CLIPVisionModel

logger = logging.getLogger(__name__)

# Type: (cluster_id, flat_index_in_cutouts, pil_image, original_meta)
RepItem = Tuple[int, int, Image.Image, tuple]


class CLIPKMeansClusterer:
    """
    1. Encode images with CLIP vision encoder.
    2. Standardise embeddings.
    3. Search for best K in [2, max_k] by silhouette score.
    4. Return the representative (nearest to centroid) image per cluster.
    """

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        max_k: int = 20,
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        self.max_k = max_k
        self.n_init = n_init
        self.random_state = random_state

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP model '%s' on %s …", clip_model, self.device)
        self.processor = CLIPImageProcessor.from_pretrained(
            clip_model, use_fast=True
        )
        self.vision_model = (
            CLIPVisionModel.from_pretrained(clip_model)
            .eval()
            .to(self.device)
        )
        logger.info("CLIP loaded.")


    def encode(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode a list of PIL images into CLIP CLS-token embeddings.

        Returns:
            Float32 array of shape (N, embed_dim).
        """
        features: List[np.ndarray] = []
        with torch.no_grad():
            for pil_img in images:
                rgb = pil_img.convert("RGB")
                inputs = self.processor(rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.vision_model(**inputs)
                cls_feat = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                features.append(cls_feat)
        return np.asarray(features, dtype=np.float32)


    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - X.mean(0)) / (X.std(0) + 1e-8)

    def fit_best_k(self, X: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        Search for the K that maximises silhouette score.

        Returns:
            labels    : Cluster label per sample.
            best_k    : Chosen number of clusters.
            best_sil  : Corresponding silhouette score.
        """
        n = len(X)
        k_range = range(2, min(self.max_k, n))
        if len(k_range) == 0:
            # Edge-case: fewer than 3 samples — put everything in 1 cluster
            logger.warning("Too few samples (%d) for silhouette search; k=1", n)
            return np.zeros(n, dtype=int), 1, 0.0

        scores: List[float] = []
        for k in k_range:
            lbl = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
            ).fit_predict(X)
            scores.append(float(silhouette_score(X, lbl)))

        best_idx = int(np.argmax(scores))
        best_k = list(k_range)[best_idx]
        best_sil = scores[best_idx]

        logger.info("Best K=%d  silhouette=%.4f", best_k, best_sil)

        kmeans = KMeans(
            n_clusters=best_k,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        labels = kmeans.fit_predict(X)
        self._kmeans = kmeans
        return labels, best_k, best_sil

    def find_representatives(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, int]:
        """
        For each cluster, find the index of the sample closest to the centroid.

        Returns:
            Dict mapping cluster_id → flat sample index.
        """
        centers = self._kmeans.cluster_centers_
        reps: Dict[int, int] = {}
        for c in range(len(centers)):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            dists = np.linalg.norm(X[idx] - centers[c], axis=1)
            reps[c] = int(idx[np.argmin(dists)])
        return reps


    def cluster(
        self,
        cutouts: List[Image.Image],
        meta: List[tuple],
    ) -> Tuple[List[RepItem], np.ndarray, int]:
        """
        Full pipeline: encode → standardise → best-K search → representatives.

        Args:
            cutouts : List of PIL RGBA images (particle cutouts).
            meta    : Parallel list of metadata tuples per cutout.

        Returns:
            rep_items : List of (cluster_id, flat_idx, pil_image, meta).
            labels    : Cluster assignment per cutout (length == len(cutouts)).
            best_k    : Number of clusters chosen.
        """
        features = self.encode(cutouts)
        X = self._standardise(features)
        labels, best_k, _ = self.fit_best_k(X)
        reps = self.find_representatives(X, labels)

        rep_items: List[RepItem] = []
        for cluster_id, img_idx in reps.items():
            rep_items.append(
                (cluster_id, img_idx, cutouts[img_idx], meta[img_idx])
            )

        return rep_items, labels, best_k
