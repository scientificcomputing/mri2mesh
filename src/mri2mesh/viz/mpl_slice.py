from __future__ import annotations
import argparse
import typing
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the file to analyze",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="results",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        default="",
        help="Tag to add to the output files",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="gray",
        help="Colormap to use",
    )
    parser.add_argument(
        "--add-colorbar",
        action="store_true",
        help="Add colorbar to the plots",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=5,
        help="Number of rows in the plot",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=5,
        help="Number of columns in the plot",
    )


def main(
    input: Path, output: Path, tag: str, cmap: str, add_colorbar: bool, nx: int, ny: int
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    img = nib.load(input).get_fdata()
    plot_slices(
        img,
        output,
        tag=tag,
        cmap=cmap,
        add_colorbar=add_colorbar,
        nx=nx,
        ny=ny,
    )
    return 0


def plot_slice(
    img: np.ndarray,
    outdir: Path | None,
    tag: str,
    cmap: str,
    labels,
    add_colorbar: bool,
    nx: int,
    ny: int,
    slice: str,
):
    minval = np.min(img)
    maxval = np.max(img)
    fig, ax = plt.subplots(nx, ny, figsize=(20, 20))

    slice2axis = {"x": 0, "y": 1, "z": 2}
    axis = slice2axis[slice]

    for k, i in enumerate(np.linspace(0, img.shape[axis], 26, dtype=int)[:-1]):
        ax.flatten()[k].set_title(f"{i}")
        img_i = img.take(indices=i, axis=axis)
        im = ax.flatten()[k].imshow(img_i, cmap=cmap, vmin=minval, vmax=maxval)
    if add_colorbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cax = fig.colorbar(im, cax=cbar_ax)
        if labels:
            cax.ax.set_yticks(np.array(list(labels.values())))
            cax.ax.set_yticklabels(np.array(list(labels.keys())))
    if outdir is not None:
        fig.savefig(outdir / f"{tag}x_slice.png")
    else:
        plt.show()
    plt.close("all")


def plot_slices(
    img: np.ndarray,
    outdir: Path | None = None,
    tag="",
    cmap="gray",
    labels=None,
    add_colorbar=False,
    nx: int = 5,
    ny: int = 5,
    slice: typing.Literal["x", "y", "z", "all"] = "all",
):
    if slice == "all":
        plot_slice(img, outdir, tag, cmap, labels, add_colorbar, nx, ny, "x")
        plot_slice(img, outdir, tag, cmap, labels, add_colorbar, nx, ny, "y")
        plot_slice(img, outdir, tag, cmap, labels, add_colorbar, nx, ny, "z")
    else:
        plot_slice(img, outdir, tag, cmap, labels, add_colorbar, nx, ny, slice)
