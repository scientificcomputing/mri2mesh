from importlib.metadata import metadata

from . import viz
from . import morphology
from . import cli
from . import vtk_utils
from . import surface
from . import mesh
from . import reader
from .reader import read

__all__ = ["viz", "morphology", "cli", "vtk_utils", "surface", "reader", "read", "mesh"]

meta = metadata("mri2mesh")
__version__ = meta["Version"]
__author__ = meta["Author-email"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]
