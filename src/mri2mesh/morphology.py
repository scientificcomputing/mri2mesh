import logging
import numpy as np
import numpy.typing as npt
import skimage.morphology as skim
import skimage
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)


def get_closest_point(a: npt.NDArray, b: np.ndarray) -> tuple[int, ...]:
    dist = ndi.distance_transform_edt(np.logical_not(a))
    assert isinstance(dist, np.ndarray)
    dist[np.logical_not(b)] = np.inf
    minidx = np.unravel_index(np.argmin(dist), a.shape)
    return minidx


def connect_by_line(
    m1: npt.NDArray, m2: npt.NDArray, footprint: npt.NDArray = skim.ball(1)
) -> npt.NDArray:
    # compute connection between V3 and V4:
    pointa = get_closest_point(m1, m2)
    pointb = get_closest_point(m2, m1)

    # add a line between the shortest points to connect V3 and V4
    line = np.array(skimage.draw.line_nd(pointa, pointb, endpoint=True))
    conn = np.zeros(m1.shape, dtype=np.uint8)
    i, j, k = line
    conn[i, j, k] = 1
    return skim.binary_dilation(conn, footprint=footprint)


def binary_smoothing(img: npt.NDArray, footprint=skim.ball(1)) -> npt.NDArray:
    openend = skim.binary_opening(img, footprint=footprint)
    return skim.binary_closing(openend, footprint=footprint)


def seperate_labels(
    img: npt.NDArray, l1: list[int], l2: list[int], dist: float, newlabel: int
) -> npt.NDArray:
    logger.debug(f"Seperating labels {l1} and {l2} with distance {dist}, new label {newlabel}")
    new_img = img.copy()
    m1 = skim.binary_dilation(img == l1, skim.ball(dist))
    m2 = skim.binary_dilation(img == l2, skim.ball(dist))
    new_img[np.logical_and(m1, m2)] = newlabel
    return new_img
