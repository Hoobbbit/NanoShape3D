#!/usr/bin/env python
"""
run_pipeline.py  —  End-to-end NanoShape3D pipeline

Steps
-----
1. Load microscopy images from disk.
2. Segment particles with Cellpose.
3. Extract per-particle RGBA cutouts.
4. Cluster cutouts with CLIP + KMeans and pick cluster representatives.
5. Generate a 3-D .glb mesh for each representative using Hunyuan3D-2.

Usage
-----
python scripts/run_pipeline.py                           # all defaults
python scripts/run_pipeline.py --image-dir my_images/   # custom input
python scripts/run_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clip_kmeans import CLIPKMeansClusterer
from src.extractor import cutouts_from_label_mask, save_cutouts
from src.cellpose_segmenter import CellposeSegmenter
from src.hunyuan_generator import ShapeGenerator
from src.io_utils import (
    ensure_dir,
    setup_logging,
    show_image_grid,
    show_segmentation_pairs,
)

logger = logging.getLogger("nanoshape3d.pipeline")


DEFAULTS = {
    "image_dir": "test_images",
    "image_ext": ".jpg",
    "results_dir": "results",
    "cellpose_model": "cellpose_sam_aug_epoch_0045",
    "cellpose_gpu": True,
    "cellpose_niter": 1000,
    "min_pixels": 100,
    "clip_model": "openai/clip-vit-large-patch14",
    "max_k": 20,
    "hunyuan_model": "tencent/Hunyuan3D-2",
    "hunyuan_subfolder": "hunyuan3d-dit-v2-0-turbo",
    "num_inference_steps": 5,
    "mc_algo": "mc",
    "seed": 12345,
    "remove_background": False,
    "save_visualisations": True,
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NanoShape3D – microscopy particles → 3D meshes"
    )
    p.add_argument("--config", type=str, help="Path to a YAML config file.")
    p.add_argument("--image-dir", type=str)
    p.add_argument("--image-ext", type=str)
    p.add_argument("--results-dir", type=str)
    p.add_argument("--cellpose-model", type=str)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--min-pixels", type=int)
    p.add_argument("--max-k", type=int)
    p.add_argument("--steps", type=int, dest="num_inference_steps")
    p.add_argument("--seed", type=int)
    p.add_argument("--skip-3d", action="store_true",
                   help="Stop after clustering (skip 3-D generation).")
    p.add_argument("--no-vis", action="store_true",
                   help="Disable saving visualisation figures.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)

    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})

    if args.image_dir:      cfg["image_dir"] = args.image_dir
    if args.image_ext:      cfg["image_ext"] = args.image_ext
    if args.results_dir:    cfg["results_dir"] = args.results_dir
    if args.cellpose_model: cfg["cellpose_model"] = args.cellpose_model
    if args.no_gpu:         cfg["cellpose_gpu"] = False
    if args.min_pixels:     cfg["min_pixels"] = args.min_pixels
    if args.max_k:          cfg["max_k"] = args.max_k
    if args.num_inference_steps: cfg["num_inference_steps"] = args.num_inference_steps
    if args.seed:           cfg["seed"] = args.seed
    if args.no_vis:         cfg["save_visualisations"] = False

    return cfg

def step_segment(cfg: dict):
    segmenter = CellposeSegmenter(
        pretrained_model=cfg["cellpose_model"],
        gpu=cfg["cellpose_gpu"],
        niter=cfg["cellpose_niter"],
    )
    imgs, files = CellposeSegmenter.load_images(
        cfg["image_dir"], extension=cfg["image_ext"]
    )
    masks, flows, styles = segmenter.segment(imgs)
    return imgs, files, masks


def step_cutouts(cfg: dict, imgs, masks):
    all_cutouts = []
    all_meta = []
    cutout_dir = ensure_dir(Path(cfg["results_dir"]) / "cutouts")

    for img_idx, (img, mask) in enumerate(zip(imgs, masks)):
        cutouts = cutouts_from_label_mask(
            img, mask,
            crop_to_mask=True,
            min_pixels=cfg["min_pixels"],
        )
        save_cutouts(cutouts, cutout_dir / f"img_{img_idx:04d}")

        for obj_id, pil_img, bbox in cutouts:
            all_cutouts.append(pil_img)
            all_meta.append((img_idx, obj_id, bbox))

    logger.info("Total cutouts across all images: %d", len(all_cutouts))
    return all_cutouts, all_meta


def step_cluster(cfg: dict, all_cutouts, all_meta):
    clusterer = CLIPKMeansClusterer(
        clip_model=cfg["clip_model"],
        max_k=cfg["max_k"],
    )
    rep_items, labels, best_k = clusterer.cluster(all_cutouts, all_meta)
    logger.info("Found %d cluster(s)", best_k)
    return rep_items, labels, best_k


def step_generate_3d(cfg: dict, rep_items):
    generator = ShapeGenerator(
        model_path=cfg["hunyuan_model"],
        subfolder=cfg["hunyuan_subfolder"],
        num_inference_steps=cfg["num_inference_steps"],
        mc_algo=cfg["mc_algo"],
        seed=cfg["seed"],
        remove_background=cfg["remove_background"],
    )

    mesh_dir = ensure_dir(Path(cfg["results_dir"]) / "meshes")
    generated = []

    for cluster_id, img_idx, pil_img, meta in rep_items:
        out_path = mesh_dir / f"cluster_{cluster_id}_rep_{img_idx}.glb"
        path = generator.generate(pil_img, output_path=out_path)
        generated.append(path)

    logger.info("Generated %d mesh(es) in %s", len(generated), mesh_dir)
    return generated


def save_vis(cfg: dict, imgs, masks, all_cutouts, rep_items):
    vis_dir = ensure_dir(Path(cfg["results_dir"]) / "visualisations")

    # Raw images
    show_image_grid(
        imgs,
        titles=[f"ID {i}" for i in range(len(imgs))],
        title="Input images",
        save_path=vis_dir / "01_input_images.png",
    )

    # Segmentation pairs
    show_segmentation_pairs(imgs, masks, save_dir=vis_dir / "02_segmentation")

    # All cutouts
    show_image_grid(
        [c for c in all_cutouts],
        titles=[f"#{i}" for i in range(len(all_cutouts))],
        title="All particle cutouts",
        save_path=vis_dir / "03_all_cutouts.png",
    )

    # Representative cutouts
    rep_imgs = [item[2] for item in rep_items]
    rep_titles = [f"Cluster {item[0]}" for item in rep_items]
    show_image_grid(
        rep_imgs,
        titles=rep_titles,
        cols=min(3, len(rep_imgs)),
        title="Cluster representatives",
        save_path=vis_dir / "04_representatives.png",
    )

    logger.info("Visualisations saved to %s", vis_dir)

def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    cfg = build_config(args)
    logger.info("=== NanoShape3D pipeline ===")
    logger.info("Config: %s", cfg)

    # Segmentation
    logger.info("--- Step 1: Segmentation ---")
    imgs, files, masks = step_segment(cfg)

    # Cutouts
    logger.info("--- Step 2: Cutout extraction ---")
    all_cutouts, all_meta = step_cutouts(cfg, imgs, masks)

    # Clustering
    logger.info("--- Step 3: Clustering ---")
    rep_items, labels, best_k = step_cluster(cfg, all_cutouts, all_meta)

    # 3D generation 
    if not args.skip_3d:
        logger.info("--- Step 4: 3-D generation ---")
        step_generate_3d(cfg, rep_items)
    else:
        logger.info("--- Step 4 skipped (--skip-3d) ---")

    # Visualisations
    if cfg["save_visualisations"]:
        save_vis(cfg, imgs, masks, all_cutouts, rep_items)

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
