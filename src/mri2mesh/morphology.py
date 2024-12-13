import logging
import numpy as np
import numpy.typing as npt
import skimage.morphology as skim
import skimage
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)


def get_closest_point(a: npt.NDArray, b: np.ndarray) -> tuple[np.int64, ...]:
    """Get the index of the closest point on `b` to `a`

    Parameters
    ----------
    a : npt.NDArray
        The object to find the closest point to
    b : np.ndarray
        The object to find the closest point on

    Returns
    -------
    tuple[int, ...]
        The index of the closest point

    Example
    -------

    >>> a = np.zeros((10, 10, 10), dtype=bool)
    >>> b = np.zeros((10, 10, 10), dtype=bool)
    >>> a[:2, :2, :2] = True
    >>> b[5:7, 5:7, 5:7] = True
    >>> a_b_index = get_closest_point(a, b)
    >>> assert np.allclose(a_b_index, (5, 5, 5))

    """
    dist = ndi.distance_transform_edt(np.logical_not(a))
    assert isinstance(dist, np.ndarray)
    dist[np.logical_not(b)] = np.inf
    minidx = np.unravel_index(np.argmin(dist), a.shape)
    return minidx


def connect_by_line(
    m1: npt.NDArray, m2: npt.NDArray, footprint: npt.NDArray = skim.ball(1)
) -> npt.NDArray:
    """Connect two objects by a line

    Parameters
    ----------
    m1 : npt.NDArray
        The first object
    m2 : npt.NDArray
        The second object
    footprint : npt.NDArray, optional
        The footprint to use for dilation, by default skim.ball(1)

    Returns
    -------
    npt.NDArray
        The connected objects

    """
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
