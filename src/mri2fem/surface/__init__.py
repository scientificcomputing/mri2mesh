import argparse
import logging
import numpy as np
import nibabel
import pyvista as pv
import skimage
import skimage.morphology as skim
from pathlib import Path

from .utils import extract_surface

# from ..image_processing import plot_slices, binary_smoothing
from ..viz import plot_slices
from ..morphology import connect_by_line, binary_smoothing
from ..constants import SYNTHSEG_LABELS, MM2M, HOLE_THRESHOLD

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the nifti file to analyze",
    )
    parser.add_argument(
        "--input-robust",
        type=Path,
        help="Path to the robust segmentation to use as mask",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default="results",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--hemisphere-min-distance",
        type=int,
        default=0,
        help="Minimum distance between hemispheres",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Padding to add to the image",
    )


def seperate_labels(img, l1, l2, dist, newlabel=SYNTHSEG_LABELS["CSF"]):
    m1 = skim.binary_dilation(img == l1, skim.ball(dist))
    m2 = skim.binary_dilation(img == l2, skim.ball(dist))
    img[np.logical_and(m1, m2)] = newlabel


def generate_par_mask(img):
    par_mask = np.logical_not(np.isin(img, SYNTHSEG_LABELS["FLUID"]))
    par_mask = skim.remove_small_objects(par_mask, HOLE_THRESHOLD)
    par_mask = skim.remove_small_holes(par_mask, HOLE_THRESHOLD)
    return par_mask


def generate_parenchyma_surface(
    outdir, img=None, resolution=(1, 1, 1), origin=(0, 0, 0), par_mask=None
):
    if par_mask is None:
        assert img is not None, "Need to provide image if par_mask is not provided"
        par_mask = generate_par_mask(img)
    plot_slices(par_mask, outdir, tag="parenchyma_mask_")
    par_surf = extract_surface(par_mask, resolution=resolution, origin=origin)
    par_surf = par_surf.smooth_taubin(n_iter=10, pass_band=0.1)
    # par_surf.points = nibabel.affines.apply_affine(seg.affine, par_surf.points)
    par_surf = par_surf.clip_closed_surface(normal=(0, 0, 1), origin=(0, 0, 1))
    par_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(f"{outdir}/parenchyma.ply", par_surf.scale(MM2M))


def generate_skull_mask(img, par_mask=None):
    if par_mask is None:
        par_mask = generate_par_mask(img)
    skull_mask = skim.remove_small_holes(img > 0, HOLE_THRESHOLD)
    skull_mask = np.logical_or(skull_mask, skim.binary_dilation(par_mask))
    skull_mask = skim.binary_dilation(skull_mask, skim.ball(1))
    skull_mask = skim.binary_dilation(binary_smoothing(skim.binary_erosion(skull_mask)))
    skull_mask[:, :, :6] = np.logical_or(
        skull_mask, skim.binary_dilation(skull_mask, skim.ball(2))
    )[:, :, :6]
    return skull_mask


def generate_skull_surface(
    outdir,
    img=None,
    resolution=(1, 1, 1),
    origin=(0, 0, 0),
    skull_mask=None,
    par_mask=None,
):
    if skull_mask is None:
        assert img is not None, "Need to provide image if skull_mask is not provided"
        skull_mask = generate_skull_mask(img, par_mask)

    plot_slices(skull_mask, outdir, tag="skull_mask_")
    skull_surf = extract_surface(skull_mask, resolution=resolution, origin=origin)
    skull_surf = skull_surf.smooth_taubin(n_iter=50, pass_band=0.1)
    # skull_surf.points = nibabel.affines.apply_affine(seg.affine, skull_surf.points)
    skull_surf = skull_surf.clip_closed_surface(normal=(0, 0, 1), origin=(0, 0, 1))
    skull_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(f"{outdir}/skull.ply", skull_surf.scale(MM2M))


def generate_cerebrum_mask(img):
    cerebrum_mask = np.isin(img, SYNTHSEG_LABELS["CEREBRUM"])
    cerebrum_mask = skim.remove_small_objects(cerebrum_mask, HOLE_THRESHOLD)
    cerebrum_mask = skim.remove_small_holes(cerebrum_mask, HOLE_THRESHOLD)
    return cerebrum_mask


def generate_cerebrum_surface(
    outdir,
    img=None,
    resolution=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    cerebrum_mask=None,
):
    if cerebrum_mask is None:
        assert img is not None, "Need to provide image if cerebrum_mask is not provided"
        cerebrum_mask = generate_cerebrum_mask(img)
    cerebrum_surf = extract_surface(cerebrum_mask, resolution=resolution, origin=origin)
    cerebrum_surf = cerebrum_surf.smooth_taubin(n_iter=10, pass_band=0.1)
    cerebrum_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(f"{outdir}/cerebrum.ply", cerebrum_surf.scale(MM2M))


def generate_ventricle_surface(
    outdir,
    ventricle_mask,
    resolution=(1, 1, 1),
    origin=(0, 0, 0),
):
    ventricle_surf = extract_surface(ventricle_mask, resolution=resolution, origin=origin)
    ventricle_surf = ventricle_surf.smooth_taubin(n_iter=20, pass_band=0.05)
    ventricle_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(f"{outdir}/ventricles.ply", ventricle_surf.scale(MM2M))


def fill_lateral_ventricls(img, outdir, resolution=(1, 1, 1), origin=(0, 0, 0)):
    new_img = img.copy()
    for LVINFID, LVID in zip(SYNTHSEG_LABELS["LV_INF"], SYNTHSEG_LABELS["LV"]):
        mask = img == LVINFID

        surf = extract_surface(mask, resolution=resolution, origin=origin)
        surf = surf.smooth_taubin(n_iter=20, pass_band=0.05)
        pv.save_meshio(f"{outdir}/inf_{LVINFID}_tmp.ply", surf.scale(MM2M))

        hull = skim.convex_hull_image(mask)
        mask = binary_smoothing(
            skim.binary_erosion(hull, skim.ball(1)) + skim.binary_dilation(mask, skim.ball(1)),
            footprint=skim.ball(2),
        )
        new_img[mask] = LVID
        surf = extract_surface(mask, resolution=resolution, origin=origin)
        surf = surf.smooth_taubin(n_iter=20, pass_band=0.05)
        pv.save_meshio(f"{outdir}/inf_{LVINFID}.ply", surf.scale(MM2M))
    return new_img


def main(input, outdir, hemisphere_min_distance, padding, input_robust):
    # outdir = args["output"]
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading segmentation from {input}")
    seg = nibabel.load(input)

    img = np.pad(seg.get_fdata(), padding)
    logger.debug(f"Loaded segmentation with shape {img.shape} and padding {padding}")

    if input_robust:
        img_rob = np.pad(nibabel.load(input_robust).get_fdata(), padding)
        plot_slices(img_rob, outdir, tag="robust_")
        img[img_rob == 0] = 0
        img[img_rob == SYNTHSEG_LABELS["BRAIN_STEM"]] = SYNTHSEG_LABELS["BRAIN_STEM"]
    resolution = np.array(seg.header["pixdim"][1:4])
    origin = -np.array(resolution) * padding

    plot_slices(img, outdir)

    # hemisphere_min_distance = args["hemisphere_min_distance"]
    if hemisphere_min_distance:
        seperate_labels(
            img,
            SYNTHSEG_LABELS["GM_LEFT"],
            SYNTHSEG_LABELS["GM_RIGHT"],
            hemisphere_min_distance,
        )
        plot_slices(img, outdir, tag="hemisphere_")

    # Generate Parenchyma surface - everything but CSF space
    par_mask = generate_par_mask(img)
    plot_slices(par_mask, outdir, tag="parenchyma_mask_")
    generate_parenchyma_surface(
        outdir=outdir, resolution=resolution, origin=origin, par_mask=par_mask
    )

    # Generate Skull surface
    skull_mask = generate_skull_mask(img=img, par_mask=par_mask)
    plot_slices(skull_mask, outdir, tag="skull_mask_")
    generate_skull_surface(
        outdir=outdir,
        resolution=resolution,
        origin=origin,
        skull_mask=skull_mask,
        par_mask=par_mask,
    )

    # Generate Cerebrum surface
    cerebrum_mask = generate_cerebrum_mask(img)
    plot_slices(cerebrum_mask, outdir, tag="cerebrum_mask_")
    generate_cerebrum_surface(
        outdir=outdir,
        resolution=resolution,
        origin=origin,
        cerebrum_mask=cerebrum_mask,
    )

    # Make inferior lateral ventricle horns
    img = fill_lateral_ventricls(img, outdir, resolution=resolution, origin=origin)

    # Connect V3 and V4 by a line
    line_V3V4 = connect_by_line(
        img == SYNTHSEG_LABELS["V3"],
        img == SYNTHSEG_LABELS["V4"],
        footprint=skim.ball(1.8),
    )
    # compute ventricle surface
    ventricle_mask = np.isin(img, SYNTHSEG_LABELS["VENTRICLE"]) + line_V3V4
    # ventricle_mask = skim.binary_dilation(ventricle_mask, footprint=skim.ball(1))
    plot_slices(ventricle_mask, outdir, tag="ventricle_mask_")
    generate_ventricle_surface(
        outdir=outdir,
        ventricle_mask=ventricle_mask,
        resolution=resolution,
        origin=origin,
    )

    # compute a layer of parenchymal tissue around the ventricles
    # to get a watertight ventricular system and
    # generate combined mask of ventricles and parenchyma

    # first find lowest point of V4 to generate an outlet of the tissue sheet into
    # the cisterna magna
    ventr_idx = np.argwhere(ventricle_mask)
    ind = np.argsort(ventr_idx[:, 2])
    lowest_ventr_point = ventr_idx[ind][0]
    ventr_outlet = np.zeros(img.shape)
    ventr_outlet[tuple(lowest_ventr_point)] = 1
    ventr_outlet = skim.binary_dilation(ventr_outlet, footprint=skim.ball(5))

    # extent ventricles to create tissue sheet and substract cisterna magna outlet
    ventricle_extended = skim.binary_dilation(ventricle_mask, footprint=skim.ball(3))
    ventricle_extended = np.logical_and(ventricle_extended, np.logical_not(ventr_outlet))

    # compute lateral ventricle surface
    LV_mask = np.isin(img, SYNTHSEG_LABELS["LV"])
    for LVID in SYNTHSEG_LABELS["LV"]:
        LV_mask += connect_by_line(
            img == LVID, img == SYNTHSEG_LABELS["V3"], footprint=skim.ball(2.5)
        )
    # LV_mask =binary_smoothing(LV_mask, footprint=skim.ball(1))
    LV_mask = skimage.filters.gaussian(LV_mask)
    plot_slices(LV_mask, outdir, tag="LV_mask_")
    LV_surf = extract_surface(LV_mask, resolution=resolution, origin=origin)
    LV_surf = LV_surf.smooth_taubin(n_iter=20, pass_band=0.05)
    LV_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(f"{outdir}/LV.ply", LV_surf.scale(MM2M))

    # compute V3 and V4 surface
    V34_mask = np.isin(img, [SYNTHSEG_LABELS["V3"], SYNTHSEG_LABELS["V4"]]) + line_V3V4
    V34_mask = skim.remove_small_objects(V34_mask, HOLE_THRESHOLD)
    V34_mask = skim.binary_dilation(V34_mask, footprint=skim.ball(1))
    plot_slices(V34_mask, outdir, tag="V34_mask_")
    V34_surf = extract_surface(V34_mask, resolution=resolution, origin=origin)
    V34_surf = V34_surf.smooth_taubin(n_iter=20, pass_band=0.05)
    V34_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(f"{outdir}/V34.ply", V34_surf.scale(MM2M))

    # finally, generate combined mask of parenchyma and ventricles
    par_ventr_mask = par_mask + ventricle_extended
    par_surf = extract_surface(par_ventr_mask, resolution=resolution, origin=origin)
    par_surf = par_surf.smooth_taubin(n_iter=20, pass_band=0.05)
    # par_surf.points = nibabel.affines.apply_affine(seg.affine, par_surf.points)
    par_surf = par_surf.clip_closed_surface(normal=(0, 0, 1), origin=(0, 0, 1))
    par_surf.compute_normals(inplace=True, flip_normals=False)
    pv.save_meshio(f"{outdir}/parenchyma_incl_ventr.ply", par_surf.scale(MM2M))
