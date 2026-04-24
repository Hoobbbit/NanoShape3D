"""
3D shape generation module.
Wraps Hunyuan3D-2 (single-view and multi-view) pipelines with optional
background removal, mesh cleaning, and texture generation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _import_hunyuan():
    try:
        from hy3dgen.shapegen import (
            DegenerateFaceRemover,
            FaceReducer,
            FloaterRemover,
            Hunyuan3DDiTFlowMatchingPipeline,
        )
        return (
            Hunyuan3DDiTFlowMatchingPipeline,
            FloaterRemover,
            DegenerateFaceRemover,
            FaceReducer,
        )
    except ImportError as e:
        raise ImportError(
            "Hunyuan3D-2 is not installed. "
            "Run: pip install -e /path/to/Hunyuan3D-2"
        ) from e


class ShapeGenerator:
    """
    Generate a cleaned 3-D mesh from a single RGBA image
    (or a front/back pair) using Hunyuan3D-2.
    """

    def __init__(
        self,
        model_path: str = "tencent/Hunyuan3D-2",
        subfolder: str = "hunyuan3d-dit-v2-0-turbo",
        use_safetensors: bool = True,
        num_inference_steps: int = 5,
        mc_algo: str = "mc",
        seed: int = 12345,
        enable_flashvdm: bool = True,
        remove_background: bool = False,
    ) -> None:
        """
        Args:
            model_path          : HuggingFace repo or local path.
            subfolder           : Sub-folder inside the repo for the DiT weights.
            use_safetensors     : Load safetensors weights.
            num_inference_steps : Diffusion steps (5 = turbo, 30 = full quality).
            mc_algo             : Marching-cubes algorithm ('mc' or 'dmc').
            seed                : Manual RNG seed for reproducibility.
            enable_flashvdm     : Enable Flash-VDM attention (faster on H100/A100).
            remove_background   : Run BackgroundRemover on RGB images before
                                  generation (set False if cutouts are already RGBA).
        """
        (
            Pipeline,
            self._FloaterRemover,
            self._DegenerateFaceRemover,
            self._FaceReducer,
        ) = _import_hunyuan()

        self.num_inference_steps = num_inference_steps
        self.mc_algo = mc_algo
        self.seed = seed
        self.remove_background = remove_background

        logger.info(
            "Loading Hunyuan3D-2 pipeline from '%s' (subfolder='%s') …",
            model_path,
            subfolder,
        )
        self.pipeline = Pipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
        )
        if enable_flashvdm:
            self.pipeline.enable_flashvdm()
        logger.info("Pipeline ready.")


    def _maybe_remove_bg(self, image: Image.Image) -> Image.Image:
        if self.remove_background and image.mode == "RGB":
            from hy3dgen.rembg import BackgroundRemover
            image = BackgroundRemover()(image)
        return image

    def _clean_mesh(self, mesh: Any) -> Any:
        mesh = self._FloaterRemover()(mesh)
        mesh = self._DegenerateFaceRemover()(mesh)
        mesh = self._FaceReducer()(mesh)
        return mesh


    def generate(
        self,
        image: Image.Image,
        output_path: Union[str, Path] = "mesh.glb",
        extra_kwargs: Optional[Dict] = None,
    ) -> Path:
        """
        Generate a 3-D mesh from a single (RGBA) image.

        Args:
            image       : Particle cutout as a PIL Image.
            output_path : Where to export the .glb file.
            extra_kwargs: Extra keyword arguments forwarded to the pipeline.

        Returns:
            Path to the exported .glb file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = self._maybe_remove_bg(image)

        kwargs = dict(
            image=image,
            num_inference_steps=self.num_inference_steps,
            mc_algo=self.mc_algo,
            generator=torch.manual_seed(self.seed),
            output_type="trimesh",
        )
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        logger.info("Running 3-D generation …")
        t0 = time.time()
        mesh = self.pipeline(**kwargs)[0]
        logger.info("Generation finished in %.1f s", time.time() - t0)

        mesh = self._clean_mesh(mesh)
        mesh.export(str(output_path))
        logger.info("Mesh saved to %s", output_path)
        return output_path

    def generate_multiview(
        self,
        front: Image.Image,
        back: Image.Image,
        output_path: Union[str, Path] = "mesh_mv.glb",
        guidance_scale: float = 7.0,
        octree_resolution: int = 380,
        extra_kwargs: Optional[Dict] = None,
    ) -> Path:
        """
        Generate a 3-D mesh from a front + back image pair.

        The pipeline used is ``Hunyuan3D-2mv``; make sure the correct
        subfolder / weights are loaded (e.g. ``hunyuan3d-dit-v2-mv-turbo``).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        front = self._maybe_remove_bg(front)
        back = self._maybe_remove_bg(back)

        kwargs = dict(
            image={"front": front, "back": back},
            num_inference_steps=self.num_inference_steps,
            octree_resolution=octree_resolution,
            guidance_scale=guidance_scale,
            mc_algo=self.mc_algo,
            generator=torch.manual_seed(self.seed),
            output_type="trimesh",
        )
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        logger.info("Running multi-view 3-D generation …")
        t0 = time.time()
        mesh = self.pipeline(**kwargs)[0]
        logger.info("Generation finished in %.1f s", time.time() - t0)

        mesh = self._clean_mesh(mesh)
        mesh.export(str(output_path))
        logger.info("Mesh saved to %s", output_path)
        return output_path

    def add_texture(
        self,
        mesh: Any,
        image: Image.Image,
        model_path: str = "tencent/Hunyuan3D-2",
        output_path: Union[str, Path] = "texture.glb",
    ) -> Path:
        """Apply texture to an already-generated mesh (requires texgen weights)."""
        output_path = Path(output_path)
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
            textured = tex_pipeline(mesh, image=image)
            textured.export(str(output_path))
            logger.info("Textured mesh saved to %s", output_path)
        except Exception as exc:
            logger.warning("Texture generation failed: %s", exc)
        return output_path
