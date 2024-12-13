import typing
import argparse
import logging

from . import parenchyma

logger = logging.getLogger(__name__)


def add_surface_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="surface-command")

    parenchyma_parser = subparsers.add_parser("parenchyma", help="Generate parenchyma surface")
    parenchyma.add_arguments(parenchyma_parser)


def dispatch(command, args: dict[str, typing.Any]) -> int:
    if command == "parenchyma":
        parenchyma.main(**args)

    else:
        raise ValueError(f"Unknown command {command}")

    return 0
