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

    tetra_mesh = meshio.Mesh(
        point_array, [("tetra", cell_array)], cell_data={"cell_tags": [marker.ravel()]}
    )

    tetra_mesh_pv = pv.from_meshio(tetra_mesh)
    pv.save_meshio(outdir / "tetra_mesh.xdmf", tetra_mesh_pv)

    np.save(outdir / "point_array.npy", point_array)
    np.save(outdir / "cell_array.npy", cell_array)
    np.save(outdir / "marker.npy", marker)

    # coords, connections = tetra.get_tracked_surfaces()
    # for i, coord in enumerate(coords):
    #     np.save(outdir / f"coords_{i}.npy", coord)

    # for i, conn in enumerate(connections):
    #     np.save(outdir / f"connections_{i}.npy", conn)

    # cell_tags.find
    # Compute incident with facets
    # Intersection exterior facets


def convert_mesh_dolfinx(
    mesh_dir: Path, extract_facet_tags: bool = False, extract_submesh: bool = False
):
    logger.info("Converting mesh to dolfinx in %s", mesh_dir)
    from mpi4py import MPI
    import dolfinx
    import basix
    import ufl

    point_array = np.load(mesh_dir / "point_array.npy")
    cell_array = np.load(mesh_dir / "cell_array.npy")
    marker = np.load(mesh_dir / "marker.npy")

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_mesh(
        comm,
        cell_array.astype(np.int64),
        point_array,
        ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
    )
    tdim = mesh.topology.dim
    fdim = tdim - 1
    local_entities, local_values = dolfinx.io.gmshio.distribute_entity_data(
        mesh,
        tdim,
        cell_array.astype(np.int64),
        marker.flatten().astype(np.int32),
    )
    adj = dolfinx.graph.adjacencylist(local_entities)
    cell_tags = dolfinx.mesh.meshtags_from_entities(
        mesh,
        tdim,
        adj,
        local_values.astype(np.int32, copy=False),
    )
    cell_tags.name = "cell_tags"
    if not extract_facet_tags:
        logger.debug("Save files")
        with dolfinx.io.XDMFFile(comm, mesh_dir / "mesh.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(cell_tags, mesh.geometry)

        return

    mesh.topology.create_connectivity(tdim - 1, tdim)

    # FIXME: Here we just add hard coded values for now. This should be fixed in the future.

    entities = []
    values = []
    # 1 = Parenchyma
    PARENCHYMA = 1

    cells = cell_tags.find(PARENCHYMA)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    incident_facets = dolfinx.mesh.compute_incident_entities(mesh.topology, cells, tdim, tdim - 1)
    exterior_facets_marker = np.intersect1d(incident_facets, exterior_facets)
    values.append(np.full(exterior_facets_marker.shape[0], PARENCHYMA, dtype=np.int32))
    entities.append(exterior_facets_marker)

    VENTRICLES = 3
    all_cell_tags = np.unique(cell_tags.values)
    cell_not_ventricles = np.setdiff1d(all_cell_tags, [VENTRICLES])
    import scifem

    interface_entities = scifem.mesh.find_interface(cell_tags, [VENTRICLES], cell_not_ventricles)
    entities.append(interface_entities)
    values.append(np.full(interface_entities.shape[0], VENTRICLES, dtype=np.int32))

    facet_tags = dolfinx.mesh.meshtags(
        mesh,
        fdim,
        np.hstack(entities),
        np.hstack(values),
    )
    facet_tags.name = "facet_tags"

    mesh.name = "mesh"

    logger.debug("Save files")
    meshname = "mesh_full.xdmf" if extract_submesh else "mesh.xdmf"
    with dolfinx.io.XDMFFile(comm, mesh_dir / meshname, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)
        xdmf.write_meshtags(cell_tags, mesh.geometry)

    logger.info("Mesh saved to %s", mesh_dir / meshname)

    if not extract_submesh:
        return
    submesh_data = scifem.mesh.extract_submesh(mesh, cell_tags, cell_not_ventricles)

    # Transfer the facet tags to the submesh
    submesh_data.domain.topology.create_connectivity(2, 3)
    facet_tags_submesh, sub_to_parent_entity_map = scifem.mesh.transfer_meshtags_to_submesh(
        facet_tags,
        # geo.facet_tags,   # If available
        submesh_data.domain,
        vertex_to_parent=submesh_data.vertex_map,
        cell_to_parent=submesh_data.cell_map,
    )

    np.save(mesh_dir / "sub_to_parent_entity_map.npy", sub_to_parent_entity_map)
    np.save(mesh_dir / "vertex_map.npy", submesh_data.vertex_map)
    np.save(mesh_dir / "cell_map.npy", submesh_data.cell_map)

    # Remove overflow values
    keep_indices = facet_tags_submesh.values > 0
    facet_tags_new = dolfinx.mesh.meshtags(
        submesh_data.domain,
        fdim,
        facet_tags_submesh.indices[keep_indices],
        facet_tags_submesh.values[keep_indices],
    )

    facet_tags_new.name = "facet_tags"
    submesh_data.cell_tag.name = "cell_tags"

    with dolfinx.io.XDMFFile(comm, mesh_dir / "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(submesh_data.domain)
        xdmf.write_meshtags(submesh_data.cell_tag, submesh_data.domain.geometry)
        xdmf.write_meshtags(facet_tags_new, submesh_data.domain.geometry)
