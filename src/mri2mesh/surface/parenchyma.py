from __future__ import annotations
import argparse
import typing
from pathlib import Path

import numpy as np
import skimage.morphology as skim
import pyvista as pv

from ..constants import HOLE_THRESHOLD
from .utils import extract_surface
from ..viz import plot_slices
from ..constants import MM2M
from ..reader import read


def generate_mask(
    img, labels: dict[str, list[int]], hole_threshold: int = HOLE_THRESHOLD
) -> np.ndarray:
    assert "FLUID" in labels, "Need to provide FLUID label"
    par_mask = np.logical_not(np.isin(img, labels["FLUID"]))
    par_mask = skim.remove_small_objects(par_mask, hole_threshold)
    par_mask = skim.remove_small_holes(par_mask, hole_threshold)
    return par_mask


def generate_surface(
    outdir,
    img=None,
    resolution=(1, 1, 1),
    origin=(0, 0, 0),
    par_mask=None,
    labels: dict[str, list[int]] | None = None,
    plot_mask: bool = False,
):
    if par_mask is None:
        assert img is not None, "Need to provide image if par_mask is not provided"
        assert labels is not None, "Need to provide labels if par_mask is not provided"
        par_mask = generate_mask(img, labels)
    if plot_mask:
        plot_slices(par_mask, outdir, tag="parenchyma_mask_")
    par_surf = extract_surface(par_mask, resolution=resolution, origin=origin)
    par_surf = par_surf.smooth_taubin(n_iter=10, pass_band=0.1)
    # par_surf.points = nibabel.affines.apply_affine(seg.affine, par_surf.points)
    par_surf = par_surf.clip_closed_surface(normal=(0, 0, 1), origin=(0, 0, 1))
    par_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(f"{outdir}/parenchyma.ply", par_surf.scale(MM2M))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the file to analyze",
    )
    parser.add_argument(
        "--input-robust",
        type=Path,
        help="Path to the robust segmentation to use as mask",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default="results",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Padding to add to the image",
    )
    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        choices=["synthseg", "neurquant"],
        default="synthseg",
        help="Labels to use for the segmentation, default is synthseg",
    )
    parser.add_argument(
        "--plot-mask",
        action="store_true",
        help="Plot slices of the mask",
    )
    parser.add_argument(
        "--hemisphere-min-distance",
        type=float,
        default=0.0,
        help="Minimum distance between hemispheres",
    )


def main(
    input: Path,
    outdir: Path,
    padding: int,
    labels: typing.Literal["synthseg", "neuroquant"] = "synthseg",
    input_robust: Path | None = None,
    plot_mask: bool = False,
    hemisphere_min_distance: float = 0.0,
) -> int:
    volume = read(input, input_robust, label_name=labels, padding=padding)
    volume.hemisphere_min_distance = hemisphere_min_distance

    outdir.mkdir(parents=True, exist_ok=True)
    generate_surface(
        outdir,
        img=volume.img,
        resolution=volume.resolution,
        origin=volume.origin,
        labels=volume.labels,
        plot_mask=plot_mask,
    )
    return 0
