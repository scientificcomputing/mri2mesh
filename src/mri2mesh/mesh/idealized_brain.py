from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def add_arguments(parser):
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
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0009,
        help="Epsilon",
    )
    parser.add_argument(
        "--edge-length-r",
        type=float,
        default=0.015,
        help="Edge length",
    )
    parser.add_argument(
        "--skip-simplify",
        action="store_true",
        help="Skip simplification",
    )
    parser.add_argument(
        "--coarsen",
        action="store_true",
        help="Coarsen",
    )
    parser.add_argument(
        "--stop-quality",
        type=int,
        default=8,
        help="Stop quality",
    )
    parser.add_argument(
        "--max-its",
        type=int,
        default=30,
        help="Max iterations",
    )
    parser.add_argument(
        "--loglevel",
        type=int,
        default=10,
        help="Log level",
    )
    parser.add_argument(
        "--disable-filtering",
        action="store_true",
        help="Disable filtering",
    )
    parser.add_argument(
        "--use-floodfill",
        action="store_true",
        help="Use floodfill",
    )
    parser.add_argument(
        "--smooth-open-boundary",
        action="store_true",
        help="Smooth open boundary",
    )
    parser.add_argument(
        "--manifold-surface",
        action="store_true",
        help="Manifold surface",
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
    epsilon: float = 0.0009,
    edge_length_r: float = 0.015,
    skip_simplify: bool = False,
    coarsen: bool = True,
    stop_quality: int = 8,
    max_its: int = 30,
    loglevel: int = 10,
    disable_filtering: bool = False,
    use_floodfill: bool = False,
    smooth_open_boundary: bool = False,
    manifold_surface: bool = False,
) -> None:
    logger.info("Generating idealized brain surface")
    from ..surface.idealized_brain import main as main_surface

    main_surface(
        outdir,
        r=r,
        parenchyma_factor=parenchyma_factor,
        lv_factor=lv_factor,
        v34_center_factor=v34_center_factor,
        v34_height_factor=v34_height_factor,
        v34_radius_factor=v34_radius_factor,
        skull_x0=skull_x0,
        skull_x1=skull_x1,
        skull_y0=skull_y0,
        skull_y1=skull_y1,
        skull_z0=skull_z0,
        skull_z1=skull_z1,
    )
    from .basic import create_mesh, CSGTree, convert_mesh_dolfinx

    csg_tree: CSGTree = {
        "operation": "union",
        "right": {
            "operation": "union",
            "left": f"{outdir}/LV.ply",
            "right": f"{outdir}/V34.ply",
        },
        "left": {
            "operation": "union",
            "left": f"{outdir}/skull.ply",
            "right": f"{outdir}/parenchyma_incl_ventr.ply",
        },
    }

    create_mesh(
        outdir,
        csg_tree=csg_tree,
        epsilon=epsilon,
        edge_length_r=edge_length_r,
        skip_simplify=skip_simplify,
        coarsen=coarsen,
        stop_quality=stop_quality,
        max_its=max_its,
        loglevel=loglevel,
        disable_filtering=disable_filtering,
        use_floodfill=use_floodfill,
        smooth_open_boundary=smooth_open_boundary,
        manifold_surface=manifold_surface,
    )
    convert_mesh_dolfinx(mesh_dir=outdir)
