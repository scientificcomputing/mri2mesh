import numpy as np
import pyvista as pv


def numpy_to_vtkImageData(img: np.ndarray) -> pv.ImageData:
    import vtk
    import vtk.util.numpy_support as numpy_support

    if img.dtype in (np.float64, np.float32):
        data_type = vtk.VTK_FLOAT
    elif img.dtype in (np.int64, np.int32):
        data_type = vtk.VTK_INT
    else:
        raise ValueError(f"Unsupported data type: {img.dtype}")

    shape = img.shape[::-1]

    flat_img_array = img.ravel()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_img_array, deep=True, array_type=data_type)

    vtk_img = vtk.vtkImageData()
    vtk_img.GetPointData().SetScalars(vtk_data)
    vtk_img.SetDimensions(*shape)
    return pv.ImageData(vtk_img)
