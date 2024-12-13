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


def volume_threshold(mesh: pv.ImageData) -> None:
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, opacity=0.1, cmap="gray", flip_scalars=True)
    plotter.add_mesh_threshold(mesh, cmap="jet", invert=False, title="Label")
    plotter.show()


def main(input: Path, axis: int = 0) -> int:
    reader = pv.get_reader(input)
    mesh = reader.read()
    volume_threshold(mesh)

    return 0
