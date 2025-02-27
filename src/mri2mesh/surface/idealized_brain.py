from pathlib import Path
import argparse
import logging
import json
import datetime
import numpy as np

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.1,
        help="Radius of the brain",
    )
    parser.add_argument(
        "--parenchyma-factor",
        type=float,
        default=0.6,
        help="Parenchyma factor",
    )
    parser.add_argument(
        "--lv-factor",
        type=float,
        default=0.2,
        help="Left ventricle factor",
    )
    parser.add_argument(
        "--v34-center-factor",
        type=float,
        default=0.4,
        help="Ventricle 3/4 center factor",
    )
    parser.add_argument(
        "--v34-height-factor",
        type=float,
        default=0.42,
        help="Ventricle 3/4 height factor",
    )
    parser.add_argument(
        "--v34-radius-factor",
        type=float,
        default=0.06,
        help="Ventricle 3/4 radius factor",
    )
    parser.add_argument(
        "--skull-x0",
        type=float,
        default=0.0103891,
        help="Skull x0",
    )
    parser.add_argument(
        "--skull-x1",
        type=float,
        default=0.155952,
        help="Skull x1",
    )
    parser.add_argument(
        "--skull-y0",
        type=float,
        default=0.0114499,
        help="Skull y0",
    )
    parser.add_argument(
        "--skull-y1",
        type=float,
        default=0.173221,
        help="Skull y1",
    )
    parser.add_argument(
        "--skull-z0",
        type=float,
        default=0.001,
        help="Skull z0",
    )
    parser.add_argument(
        "--skull-z1",
        type=float,
        default=0.154949,
        help="Skull z1",
    )


def main(
    outdir: Path,
    r: float = 0.1,
    parenchyma_factor: float = 0.6,
    lv_factor: float = 0.2,
    v34_center_factor: float = 0.4,
    v34_height_factor: float = 0.42,
    v34_radius_factor: float = 0.06,
    skull_x0: float = 0.0103891,
    skull_x1: float = 0.155952,
    skull_y0: float = 0.0114499,
    skull_y1: float = 0.173221,
    skull_z0: float = 0.001,
    skull_z1: float = 0.154949,
) -> None:
    import pyvista as pv

    logger.info("Creating idealized brain surface in %s", outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    z = np.array([0, 0, 1])
    skull = pv.Box((skull_x0, skull_x1, skull_y0, skull_y1, skull_z0, skull_z1))
    c = skull.center_of_mass()
    par = pv.Sphere(r * parenchyma_factor, center=c)
    LV = pv.Sphere(r * lv_factor, center=c)
    V34 = pv.Cylinder(
        c - v34_center_factor * r * z,
        direction=z,
        height=r * v34_height_factor,
        radius=v34_radius_factor * r,
    )
    ventricles = pv.merge([LV, V34])

    for s, n in zip(
        [V34, LV, par, skull, ventricles],
        ["V34", "LV", "parenchyma_incl_ventr", "skull", "ventricles"],
    ):
        pv.save_meshio(outdir / f"{n}.ply", s)

    from .. import __version__

    (outdir / "surface_parameters.json").write_text(
        json.dumps(
            {
                "r": r,
                "parenchyma_factor": parenchyma_factor,
                "lv_factor": lv_factor,
                "v34_center_factor": v34_center_factor,
                "v34_height_factor": v34_height_factor,
                "v34_radius_factor": v34_radius_factor,
                "skull_x0": skull_x0,
                "skull_x1": skull_x1,
                "skull_y0": skull_y0,
                "skull_y1": skull_y1,
                "skull_z0": skull_z0,
                "skull_z1": skull_z1,
                "version": __version__,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            indent=2,
        )
    )
