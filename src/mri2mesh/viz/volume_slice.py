import argparse
from pathlib import Path
import pyvista as pv

index2title = {0: "x", 1: "y", 2: "z"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the file to analyze",
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=int,
        default=0,
        help="Axis to slice along, 0=x, 1=y, 2=z 3=all (no slider)",
    )
    parser.add_argument(
        "--slider",
        action="store_true",
        help="Add a slider to the plot",
    )


def volume_slice_all(mesh: pv.ImageData) -> None:
    p = pv.Plotter(shape=(1, 3), border=False)
    p.add_mesh(mesh.outline())

    for axis in range(3):
        p.subplot(0, axis)
        p.add_title(f"{index2title[axis]}-coordinate")
        p.add_mesh(mesh.outline())
        normal = [0, 0, 0]
        normal[axis] = 1
        origin = list(mesh.origin)
        origin[axis] = mesh.dimensions[axis] // 2
        p.add_mesh_slice(
            mesh,
            name="slice",
            normal=normal,
            origin=origin,
        )

    p.show()


def volume_slice(mesh: pv.ImageData, axis=0, slider: bool = False) -> None:
    assert axis in [0, 1, 2], f"Invalid axis {axis}"

    plotter = pv.Plotter()
    plotter.add_mesh(mesh.outline())
    plotter.add_title(f"{index2title[axis]}-coordinate")
    origin = list(mesh.origin)
    normal = [0, 0, 0]
    normal[axis] = 1
    origin[axis] = mesh.dimensions[axis] // 2

    if slider:

        def update(value):
            origin[axis] = value
            plotter.add_mesh_slice(mesh, name="slice", normal=normal, origin=origin)

        plotter.add_slider_widget(
            update, [0, mesh.dimensions[axis]], title=f"{index2title[axis]}-coordinate"
        )

    else:
        plotter.add_mesh_slice(
            mesh,
            name="slice",
            normal=normal,
            origin=origin,
        )

    plotter.show()


def main(input: Path, axis: int = 3, slider: bool = False) -> int:
    reader = pv.get_reader(input)
    mesh = reader.read()
    if axis == 3:
        assert not slider, "Slider not supported for all axes"
        volume_slice_all(mesh)
    else:
        volume_slice(mesh, axis=axis, slider=slider)

    return 0
