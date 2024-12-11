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
    plotter.add_mesh(mesh, opacity=0.1)
    plotter.add_mesh_threshold(mesh)

    # plotter.add_mesh(mesh.outline())
    # origin = list(mesh.origin)
    # normal = [0, 0, 0]
    # normal[axis] = 1
    # origin[axis] = mesh.dimensions[axis] // 2

    # if slider:

    #     def update(value):
    #         origin[axis] = value
    #         plotter.add_mesh_slice(mesh, name="slice", normal=normal, origin=origin)

    #     plotter.add_slider_widget(update, [0, mesh.dimensions[axis]], title="x-coordinate")

    # else:
    #     plotter.add_mesh_slice(mesh, name="slice", normal=normal, origin=origin)

    plotter.show()


def main(input: Path, axis: int = 0) -> int:
    reader = pv.get_reader(input)
    mesh = reader.read()
    volume_threshold(mesh)

    return 0
