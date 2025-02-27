import pytest

from mri2mesh import cli


@pytest.mark.parametrize("name", ["synthseg", "neuroquant"])
def test_labels_cli(name, capsys):
    cli.main(["labels", name])
    captured = capsys.readouterr()
    # Both labels have Background as the first label
    assert captured.out.startswith("BACKGROUND: [0]")


def test_mesh_idealized_cli(tmp_path):
    cli.main(["mesh", "idealized", "-o", str(tmp_path)])
    for name in [
        "connections_3.npy",
        "coords_0.npy",
        "coords_1.npy",
        "connections_2.npy",
        "connections_0.npy",
        "coords_3.npy",
        "coords_2.npy",
        "connections_1.npy",
        "mesh.h5",
        "ventricles.ply",
        "skull.ply",
        "tetra_mesh.xdmf",
        "mesh.xdmf",
        "LV.ply",
        "surface_parameters.json",
        "tetra_mesh.h5",
        "parenchyma_incl_ventr.ply",
        "V34.ply",
        "mesh_params.json",
    ]:
        assert (tmp_path / name).exists()
