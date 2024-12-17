import pyvista as pv


def extract_surface(img, resolution=(1, 1, 1), origin=(0, 0, 0)):
    # img should be a binary 3D np.array
    grid = pv.ImageData(dimensions=img.shape, spacing=resolution, origin=origin)
    mesh = grid.contour(isosurfaces=[0.5], scalars=img.flatten(order="F"), method="marching_cubes")
    surf = mesh.extract_geometry()
    surf.clear_data()
    return surf
