# pyvista-ftetwild-pipeline

This repository contains a pipeline to generate surfaces from voxelized data using `scikit-image` and `pyvista`. It also contains tools for visualization using `pyvista`.

## Installation

To install the required packages, run:

```bash
python3 -m pip install git+https://github.com/finsberg/pyvista-ftetwild-pipeline.git
```

## Usage
The basic using is through the command line using the command `mri2fem`. To see all the options, run:

```bash
mri2fem --help
```

### Visualization
Visualization is achieved through the subcommand `viz`. To see all options you can do

```bash
mri2fem viz --help
```

For example to visualize a nifty file called `T1_synthseg.nii.gz`, run:

```bash
mri2fem viz volume-clip -i T1_synthseg.nii.gz
```
which will open up the volume with a clipping plane. To see all the options, run:

```bash
mri2fem viz volume-clip --help
```

### Surface generation
To generate a surface from a nifty file, run:

```bash
mri2fem surface -i T1_synthseg.nii.gz
```

## Authors
The pipeline is developed by Marius Causemann and Henrik Finsberg.


## License
MIT
