import argparse
from pathlib import Path
import pyvista as pv


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the file to analyze",
    )


def volume_clip(mesh: pv.ImageData) -> None:
    p = pv.Plotter()
    p.add_mesh_clip_plane(mesh)
    p.show()


def main(input: Path) -> int:
    reader = pv.get_reader(input)
    mesh = reader.read()
    volume_clip(mesh)

    return 0
