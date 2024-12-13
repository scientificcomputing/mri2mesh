import logging
import argparse

from . import viz, surface


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Root parser
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print the command and do not run it",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print more information",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Visualization parser
    viz_parser = subparsers.add_parser("viz", help="Visualize data")
    viz.add_viz_parser(viz_parser)

    # Surface generation parser
    surface_parser = subparsers.add_parser("surface", help="Generate surfaces")
    surface.add_surface_parser(surface_parser)
    return parser


def _disable_loggers():
    for libname in ["matplotlib"]:
        logging.getLogger(libname).setLevel(logging.WARNING)


def dispatch(parser: argparse.ArgumentParser) -> int:
    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.DEBUG if args.pop("verbose") else logging.INFO)
    _disable_loggers()

    logger = logging.getLogger(__name__)
    dry_run = args.pop("dry_run")
    command = args.pop("command")

    if dry_run:
        logger.info("Dry run: %s", command)
        logger.info("Arguments: %s", args)
        return 0

    try:
        if command == "viz":
            viz.dispatch(args.pop("viz-command"), args)
        elif command == "surface":
            surface.dispatch(args.pop("surface-command"), args)
        else:
            logger.error(f"Unknown command {command}")
            parser.print_help()
    except ValueError as e:
        logger.error(e)
        parser.print_help()

    return 0


def main() -> int:
    parser = setup_parser()
    return dispatch(parser)
