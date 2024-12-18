import argparse
import typing

from . import volume_clip, volume_slice, mpl_slice, volume_threshold
from .mpl_slice import plot_slices

__all__ = ["add_viz_parser", "dispatch", "numpy_to_vtkImageData", "plot_slices", "mpl_slice"]


def list_viz_commands() -> list[str]:
    return ["volume-clip", "volume-slice", "volume-threshold", "mpl-slice"]


def add_viz_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="viz-command")

    volume_clip_parser = subparsers.add_parser("volume-clip", help="Clip a volume")
    volume_clip.add_arguments(volume_clip_parser)

    volume_slice_parser = subparsers.add_parser("volume-slice", help="Slice a volume")
    volume_slice.add_arguments(volume_slice_parser)

    volume_threshold_parser = subparsers.add_parser("volume-threshold", help="Threshold a volume")
    volume_threshold.add_arguments(volume_threshold_parser)

    mpl_slice_parser = subparsers.add_parser("mpl-slice", help="Plot slices using matplotlib")
    mpl_slice.add_arguments(mpl_slice_parser)


def dispatch(command, args: dict[str, typing.Any]) -> int:
    if command == "volume-clip":
        volume_clip.main(**args)

    elif command == "volume-slice":
        volume_slice.main(**args)

    elif command == "volume-threshold":
        volume_threshold.main(**args)

    elif command == "mpl-slice":
        mpl_slice.main(**args)

    else:
        raise ValueError(f"Unknown command {command}")

    return 0
