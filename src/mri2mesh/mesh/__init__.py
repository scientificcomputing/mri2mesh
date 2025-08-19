import typing
import logging
import argparse
from pathlib import Path

from . import basic, idealized_brain
from .basic import create_mesh

__all__ = ["basic", "idealized_brain", "create_mesh"]

logger = logging.getLogger(__name__)


def add_mesh_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="mesh-command")

    template_parser = subparsers.add_parser(
        "template", help="Create a sample configuration file called"
    )
    template_parser.add_argument("outdir", type=Path, help="Output directory")
    template_parser.add_argument(
        "--name", type=str, default="config.json", help="Name of the configuration file"
    )

    create_parser = subparsers.add_parser("create", help="Create a mesh")
    create_parser.add_argument("filename", type=Path, help="Path to the configuration file")

    convert_parser = subparsers.add_parser("convert", help="Convert mesh to dolfinx")
    convert_parser.add_argument("mesh_dir", type=Path, help="Directory containing mesh files")
    convert_parser.add_argument(
        "--extract-facet-tags", action="store_true", help="Extract facet tags"
    )
    convert_parser.add_argument("--extract-submesh", action="store_true", help="Extract submesh")

    idealized_parser = subparsers.add_parser(
        "idealized",
        help="Generate idealized surface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    idealized_brain.add_arguments(idealized_parser)


def dispatch(command, args: dict[str, typing.Any]) -> int:
    if command == "template":
        basic.generate_sameple_config(**args)

    elif command == "create":
        basic.create_mesh_from_config(**args)
    elif command == "convert":
        basic.convert_mesh_dolfinx(**args)

    elif command == "idealized":
        idealized_brain.main(**args)

    else:
        raise ValueError(f"Unknown command {command}")

    return 0
