import typing
import json
import logging
import pprint
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def sample_config():
    tree = {
        "operation": "union",
        "right": {
            "operation": "union",
            "left": "surfaces/LV.ply",
            "right": "surfaces/V34.ply",
        },
        "left": {
            "operation": "union",
            "left": "surfaces/skull.ply",
            "right": "surfaces/parenchyma_incl_ventr.ply",
        },
    }

    config = {
        "outdir": "mesh",
        "epsilon": 0.0009,
        "edge_length_r": 0.015,
        "coarsen": True,
        "stop_quality": 8,
        "max_its": 30,
        "loglevel": 10,
        "disable_filtering": False,
        "use_floodfill": False,
        "smooth_open_boundary": False,
        "manifold_surface": False,
        "csg_tree": tree,
    }
    return config


def default_config() -> dict[str, typing.Any]:
    config = sample_config()
    config["csg_tree"] = {}
    return config


def generate_sameple_config(outdir: Path, name: str = "config.json"):
    logger.info("Generating sample configuration file")
    config = sample_config()
    path = outdir / name
    if path.suffix == ".json":
        path.write_text(json.dumps(config, indent=2))
    elif path.suffix == ".toml":
        import toml

        path.write_text(toml.dumps(config))
    elif path.suffix == ".yaml":
        import yaml

        path.write_text(yaml.dump(config))
    else:
        raise ValueError(f"Unknown file extension {path.suffix}")
    logger.info("Configuration file written to %s", path)


def read_config(path: Path) -> dict[str, typing.Any]:
    logger.info("Reading configuration file %s", path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    elif path.suffix == ".toml":
        import toml

        return toml.loads(path.read_text())
    elif path.suffix == ".yaml":
        import yaml

        return yaml.load(path.read_text(), Loader=yaml.FullLoader)
    else:
        raise ValueError(f"Unknown file extension {path.suffix}")


class CSGTree(typing.TypedDict):
    operation: str
    left: typing.Union[str, "CSGTree"]
    right: typing.Union[str, "CSGTree"]


def create_mesh_from_config(filename: Path) -> Path:
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} not found")
    config = default_config()
    config.update(read_config(filename))
    create_mesh(**config)
    return Path(config["outdir"])


def create_mesh(
    outdir: typing.Union[Path, str],
    csg_tree: CSGTree,
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
):
    logger.info("Creating mesh, with the following configuration")
    params = {
        "outdir": str(outdir),
        "csg_tree": csg_tree,
        "epsilon": epsilon,
        "edge_length_r": edge_length_r,
        "skip_simplify": skip_simplify,
        "coarsen": coarsen,
        "stop_quality": stop_quality,
        "max_its": max_its,
        "loglevel": loglevel,
        "disable_filtering": disable_filtering,
        "use_floodfill": use_floodfill,
        "smooth_open_boundary": smooth_open_boundary,
        "manifold_surface": manifold_surface,
    }
    logger.info(pprint.pformat(params))

    import wildmeshing as wm
    import meshio
    import pyvista as pv

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "mesh_params.json").write_text(json.dumps(params, indent=2))

    tetra = wm.Tetrahedralizer(
        epsilon=epsilon,
        edge_length_r=edge_length_r,
        coarsen=coarsen,
        stop_quality=stop_quality,
        max_its=max_its,
        skip_simplify=skip_simplify,
    )
    tetra.set_log_level(loglevel)

    tetra.load_csg_tree(json.dumps(csg_tree))
    tetra.tetrahedralize()
    point_array, cell_array, marker = tetra.get_tet_mesh(
        all_mesh=disable_filtering,
        smooth_open_boundary=smooth_open_boundary,
        floodfill=use_floodfill,
        manifold_surface=manifold_surface,
        correct_surface_orientation=True,
    )
    coords, connections = tetra.get_tracked_surfaces()

    tetra_mesh = meshio.Mesh(
        point_array, [("tetra", cell_array)], cell_data={"cell_tags": [marker.ravel()]}
    )
    tetra_mesh_pv = pv.from_meshio(tetra_mesh).clean()
    pv.save_meshio(outdir / "tetra_mesh.xdmf", tetra_mesh_pv)

    for i, coord in enumerate(coords):
        np.save(outdir / f"coords_{i}.npy", coord)

    for i, conn in enumerate(connections):
        np.save(outdir / f"connections_{i}.npy", conn)


def convert_mesh_dolfinx(mesh_dir: Path):
    logger.info("Converting mesh to dolfinx in %s", mesh_dir)
    from mpi4py import MPI
    from scipy.spatial.distance import cdist
    import dolfinx

    threshold = 1.0
    fdim = 2

    coords = []
    for path in sorted(mesh_dir.glob("coords_*.npy"), key=lambda x: int(x.stem.split("_")[-1])):
        coords.append(np.load(path))
    logger.debug(f"Found {len(coords)} coordinates")

    connections = []
    for path in sorted(
        mesh_dir.glob("connections_*.npy"), key=lambda x: int(x.stem.split("_")[-1])
    ):
        connections.append(np.load(path))
    logger.debug(f"Found {len(connections)} connections")

    assert len(connections) == len(coords)

    logger.debug("Loading mesh")
    comm = MPI.COMM_WORLD
    with dolfinx.io.XDMFFile(comm, mesh_dir / "tetra_mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")

    logger.debug("Mesh loaded")

    facets = []
    values = []
    for i, coord in enumerate(coords, start=1):
        logger.debug(f"Processing coord {i}")

        def locator(x):
            # Find the distance to all coordinates
            distances = cdist(x.T, coord)
            # And return True is they are close
            return np.any(distances < threshold, axis=1)

        f = dolfinx.mesh.locate_entities_boundary(mesh, dim=fdim, marker=locator)
        v = np.full(f.shape[0], i, dtype=np.int32)
        facets.append(f)
        values.append(v)

    logger.debug("Create meshtags")
    facet_tags = dolfinx.mesh.meshtags(
        mesh,
        fdim,
        np.hstack(facets),
        np.hstack(values),
    )
    facet_tags.name = "facet_tags"
    cell_tags.name = "cell_tags"
    mesh.name = "mesh"

    logger.debug("Save files")
    with dolfinx.io.XDMFFile(comm, mesh_dir / "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)
        xdmf.write_meshtags(cell_tags, mesh.geometry)

    logger.info("Mesh saved to %s", mesh_dir / "mesh.xdmf")
